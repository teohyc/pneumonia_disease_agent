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

from io import BytesIO

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
    search_kwargs={"k": 6}
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
            outputs = vit_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        #probabilities for all classes and transfer to cpu and to numpy array
        probs = probabilities[0].cpu().numpy()
        
        #list of class_name, probability and sort by probability descending
        class_probs = [(class_names[i], float(probs[i])) for i in range(len(class_names))]
        class_probs.sort(key=lambda x: x[1], reverse=True)
        
        #top predicted class and confidence
        predicted_class = class_probs[0][0]
        confidence_score = class_probs[0][1]

        torch.cuda.empty_cache() #clear GPU memory
        
        #top_k list
        top_k = [{"class": class_name, "prob": round(prob, 4)} for class_name, prob in class_probs]
        
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence_score, 4),
            "top_k": top_k
        }
        
    except Exception as e:
        return {
            "predicted_class": "Error",
            "confidence": 0.0,
            "top_k": [],
            "error": str(e)
        }
    
#declare RAG retriever tool
@tool
def retriever_tool(query: str) -> list:
    """
    Retrieve relevant documents from vector store based on query.
    """
    results = retriever.invoke(query)
    return [result.page_content for result in results]

#define ViT node
def vit_node(state: AgentState) -> AgentState:
    vit_result = vit_inference.invoke({"image_path": state["image_path"]})
    
    return {
        "vit_result": vit_result,
        "retrieved_docs": [],
        "advisor_feedback": "",
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

    return {"retrieved_docs": docs}

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
"""        
    )

    response = reasoner_llm.invoke([prompt])

    print(response.content) #debugging

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
While you may use the performance metrics to guide your explanation, NEVER MENTION THE PERFORMANCE METRICS OR IMPLY THEM IN YOUR EXPLANATION.

If supporting documents are limited, explicitly state that evidence is limited.

Provide structured explanation:

1. Diagnosis (include predicted class and confidence)
2. Supporting Evidence
3. Uncertainty Discussion

Base everything strictly on the ViT output and retrieved documents.
"""       
    )

    response = explainer_llm.invoke([prompt])

    return {"messages": [response]}

#loop control
def should_loop(state: AgentState):
    if state["iteration"] >= state["max_iterations"]:
        
        print("explain")
        print(f"iteration: {state["iteration"]}") #debugging

        return "explain"
    
    decision_text = state["advisor_feedback"].lower()
    if "decision: insufficient" in decision_text:

        print("loop")
        print(f"iteration: {state["iteration"]}")

        return "loop"
    
    print(f"iteration: {state["iteration"]}")
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

""" #drawing the flow diagram
png = app.get_graph().draw_mermaid_png()
img = Image.open(BytesIO(png))
img.show()
img.save("pneumonia_agent_diagram.png")
"""

def run_agent(image_path: str, max_iterations: int = 3):
     initial_state = {
        "messages": [HumanMessage(content="Analyze this chest X-ray image.")],
        "vit_result": {},
        "retrieved_docs": [],
        "image_path": image_path,
        "iteration": 0,
        "max_iterations": max_iterations,
        "advisor_feedback": ""
     }

     result = app.invoke(initial_state)

     print("\n=======================")
     print("FINAL EXPLANATION")
     print("=======================\n")

     print(result["messages"][-1].content)

#main
if __name__ == "__main__":

    image_path = input("Enter path to chest X-ray image").strip()

    try:
        run_agent(image_path=image_path, max_iterations=3)
    except Exception as e:
        print(f"\nError running agent: {e}")