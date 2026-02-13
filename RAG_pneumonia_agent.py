from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Sequence
from langchain_chroma import Chroma
import torch
from PIL import Image
import torchvision.transforms as transforms
from ViT_model import ViT

#visualisation 
from io import BytesIO
import cv2
import numpy as np

#declare vectorstore and retriever for RAG
PERSIST_DIR = "vectordb"
COLLECTION_NAME = "chest_xray_knowledge"

embeddings = OllamaEmbeddings(
    model="embeddinggemma:latest"
)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

#declare llm used
reasoner_llm = ChatOllama(
model="gemma3:latest",
    temperature=0.0
)

explainer_llm = ChatOllama(
    model="gemma3:latest",
    temperature=0.0
)

#load Diffusion-Augmented ViT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = ViT()
vit_model.load_state_dict(torch.load("vit_with_generated.pth", map_location=device))
vit_model = vit_model.to(device)
vit_model.eval()

#declaring agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    vit_result: dict
    retrieved_docs: list
    image_path: str
    heatmap_path: str

    iteration: int
    max_iterations: int #max iterations for reasoner-RAG retriever loop to run

    advisor_feedback: str #previous advice from reasoner on querying


#declare Diffusion-Augmented ViT tool node
@tool
def vit_inference(image_path: str) -> dict:
    """
    Run ViT on chest X-ray and return prediction.
    """

    class_names = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia", "COVID-19"]

    try:
        # load and preprocess image
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        
        # define preprocessing transform
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        #inference
        with torch.no_grad():
            outputs, attn_weights_all = vit_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        #probabilities for all classes and transfer to cpu and to numpy array
        probs = probabilities[0].cpu().numpy()
        
        #list of class_name, probability and sort by probability descending
        class_probs = [(class_names[i], float(probs[i])) for i in range(len(class_names))]
        class_probs.sort(key=lambda x: x[1], reverse=True)
        
        #top predicted class and confidence
        predicted_class = class_probs[0][0]
        confidence_score = class_probs[0][1]

        #For heatmap
        # (B, heads, 65, 65)
        attn = attn_weights_all[-1][0]  # first image

        #average across heads 
        attn = attn.mean(dim=0)  # (65, 65)

        #CLS token attention to patches 
        cls_attn = attn[0, 1:]  # (64,)

        #reshape to 8x8 grid
        heatmap = cls_attn.reshape(8, 8).cpu().numpy()

        #normalize 
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        #resize
        original = Image.open(image_path).convert("RGB")
        original_np = np.array(original).astype(np.float32) / 255.0

        heatmap = cv2.resize(
            heatmap,
            (original_np.shape[1], original_np.shape[0])
        )

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        overlay = 0.5 * heatmap + 0.5 * original_np
        overlay = np.uint8(255 * overlay)

        output_path = "attention_overlay.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        torch.cuda.empty_cache() #clear GPU memory
        
        #top_k list
        top_k = [{"class": class_name, "prob": round(prob, 4)} for class_name, prob in class_probs]
        
        print(predicted_class, round(confidence_score, 4)) #debugging

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence_score, 4),
            "top_k": top_k
        }, output_path
        
    except Exception as e:
        return {
            "predicted_class": "Error",
            "confidence": 0.0,
            "top_k": [],
            "error": str(e)
        }, output_path
    
#declare RAG retriever tool
@tool
def retriever_tool(query: str) -> list:
    """
    Retrieve relevant documents from vector store based on query.
    """
    results = retriever.invoke(query)
    return [result.page_content for result in results]

#utility for gradcam
def reshape_transform(tensor, height=8, width=8):
    # remove CLS token
    tensor = tensor[:, 1:, :]

    # reshape into spatial map
    result = tensor.reshape(tensor.size(0), 8, 8, tensor.size(2))

    # convert to (B, C, H, W)
    result = result.permute(0, 3, 1, 2)
    return result

#define ViT node
def vit_node(state: AgentState) -> AgentState:
    vit_result, heatmap_path = vit_inference.invoke({"image_path": state["image_path"]})
    
    return {
        "vit_result": vit_result,
        "retrieved_docs": [],
        "advisor_feedback": "",
        "heatmap_path": heatmap_path
    }

#define reasoner 1 node, query formulater
def reasoner_query_node(state: AgentState) -> AgentState:
    vit = state["vit_result"]

    #prompt for reasoner to formulate retrieval query based on ViT output
    prompt = SystemMessage(
        content=f"""
You are an X-Ray medical reasoning agent.
Do not fabricate any information.

A vision transformer (ViT) model has analyzed a chest X-ray image and provided the following output:
ViT Output:
Class: {vit['predicted_class']}
Confidence: {vit['confidence']}
Top Alternatives: {vit['top_k']}

Previously retrieved documents:
==================START OF PREVIOUS DOCUMENTS==================
{state['retrieved_docs'] if state['retrieved_docs'] else "None"}
+==================END OF PREVIOUS DOCUMENTS==================

Advisor feedback from evidence evaluator:
{state['advisor_feedback'] if state['advisor_feedback'] else "None"}

Formulate ONE precise medical retrieval query.
Return ONLY the query text.
"""                            
    )

    response = reasoner_llm.invoke([prompt])

    return {"messages": [response]}

#define RAG retrieval node'
def retrieval_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    docs = retriever_tool.invoke({"query": query})

    # ensure we accumulate rather than overwrite previous docs
    prev = state.get("retrieved_docs", []) or []
    combined = prev.copy()
    for d in docs:
        if d not in combined:
            combined.append(d)

    return {"retrieved_docs": combined}

#define reasoner 2 node, evidence evaluator
def reasoner_evidence_node(state: AgentState) -> AgentState:
    iteration = state["iteration"] + 1
    vit = state["vit_result"]

    prompt = SystemMessage(
        content=f"""
You are evaluating medical evidence.
Do not fabricate any information.

A vision transformer (ViT) model has analyzed a chest X-ray image and provided the following output:
ViT Result:
Class: {vit['predicted_class']}
Confidence: {vit['confidence']}
Top Alternatives: {vit['top_k']}

Retrieved Documents:
{state['retrieved_docs']}

Output STRICTLY in this format:

DECISION: SUFFICIENT or INSUFFICIENT 
REASON: <brief reasoning>
ADVICE: <what the query formulator should search next if insufficient>

(Output "DECISION: INSUFFICIENT" if the documents are not enough or the Confidence of ViT result is less than 90%)
"""        
    )

    response = reasoner_llm.invoke([prompt])

    print(response.content) #debugging
    print(len(state["retrieved_docs"]))

    return {
        "messages": [response],
        "iteration": iteration,
        "advisor_feedback": response.content
    }

#explainer node
def explainer_node(state: AgentState) -> AgentState:
    
    vit = state["vit_result"]
    prompt = SystemMessage(
        content=f"""
You are an X-Ray medical explanation agent.

A vision transformer (ViT) model has analyzed a chest X-ray image and provided the following output:
ViT Prediction:
Class: {vit['predicted_class']}
Confidence: {vit['confidence']}
Top Alternatives: {vit['top_k']}

Supporting Documents:
{state['retrieved_docs']}

CONFIDENTIAL INFORMATION:
The ViT model has the following classification performance metrics:
================================================
                precision   recall  f1-score  

      Normal       0.93      0.98      0.96       
   Bacterial       0.79      0.86      0.82       
       Viral       0.74      0.54      0.62       
    COVID-19       0.97      0.97      0.97       

    accuracy                           0.85      
   macro avg       0.86      0.84      0.84      
weighted avg       0.85      0.85      0.85      
================================================
While you may use the performance metrics to guide your explanation, NEVER MENTION OR DISCUSS THE PERFORMANCE METRICS OR IMPLY THEM IN YOUR EXPLANATION AS THE TABLE PROVIDED ABOVE IS CONFIDENTIAL.

If supporting documents are limited, explicitly state that evidence is limited.

Provide structured explanation:

1. Present the X-ray ViT diagnosis result professionally in a table form easy to be read with the confidence score of all classes presented.
2. Diagnosis (include predicted class and confidence)
3. Supporting Evidence
4. Uncertainty Discussion

Base everything strictly on the ViT output and retrieved documents. Always explain professionally, intelligently  
"""       
    )

    response = explainer_llm.invoke([prompt])

    return {"messages": [response]}
'''
#gradcam node
def gradcam_node(state: AgentState) -> AgentState:

    image_path = state["image_path"]

    
    # ---- Use LAST LAYER attention ----
    # shape: (B, heads, 65, 65)
    attn = attn_weights_all[-1][0]  # first image

    # ---- Average across heads (or try max for sharper) ----
    attn = attn.mean(dim=0)  # (65, 65)

    # ---- CLS token attention to patches ----
    cls_attn = attn[0, 1:]  # (64,)

    # ---- Reshape to 8x8 grid ----
    heatmap = cls_attn.reshape(8, 8).cpu().numpy()

    # ---- Normalize properly ----
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # ---- Resize to image size ----
    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original).astype(np.float32) / 255.0

    heatmap = cv2.resize(
        heatmap,
        (original_np.shape[1], original_np.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = 0.5 * heatmap + 0.5 * original_np
    overlay = np.uint8(255 * overlay)

    output_path = "attention_overlay.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {"heatmap_path": output_path}
'''
#loop control
def should_loop(state: AgentState):
    if state["iteration"] >= state["max_iterations"]:
        
        print("explain")
        print(f"""iteration: {state["iteration"]}""") #debugging

        return "explain"
    
    decision_text = state["advisor_feedback"].lower()
    if "decision: insufficient" in decision_text:

        print("loop")
        print(f"""iteration: {state["iteration"]}""")

        return "loop"
    
    print(f"""iteration: {state["iteration"]}""")
    print("explain")

    return "explain"

#### BUILD GRAPH ####

graph = StateGraph(AgentState)

graph.add_node("vit", vit_node)
graph.add_node("reason_query", reasoner_query_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("reason_evidence", reasoner_evidence_node)
graph.add_node("explain", explainer_node)

graph.set_entry_point("vit")

graph.add_edge("vit", "reason_query")
graph.add_edge("reason_query", "retrieve")
graph.add_edge("retrieve", "reason_evidence")

graph.add_conditional_edges(
    "reason_evidence",
    should_loop,
    {
        "loop": "reason_query",
        "explain": "explain"
    }
)

graph.add_edge("explain", END)

app = graph.compile()

"""#drawing the flow diagram
png = app.get_graph().draw_mermaid_png()
img = Image.open(BytesIO(png))
img.show()
img.save("pneumonia_agent_diagram.png")

"""
def run_agent(image_path: str, max_iterations: int = 6):
     initial_state = {
        "messages": [HumanMessage(content="Analyze this chest X-ray image.")],
        "vit_result": {},
        "retrieved_docs": [],
        "image_path": image_path,
        "heatmap_path": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "advisor_feedback": ""
     }

     result = app.invoke(initial_state)

     print("\n=======================")
     print("FINAL EXPLANATION")
     print("=======================\n")

     print(result["messages"][-1].content)

     print("Grad-CAM overlay saved at:")
     print(result["heatmap_path"])

#main
if __name__ == "__main__":

    image_path = input("Enter path to chest X-ray image: ").strip()

    try:
        run_agent(image_path=image_path, max_iterations=6)
    except Exception as e:
        print(f"\nError running agent: {e}")
