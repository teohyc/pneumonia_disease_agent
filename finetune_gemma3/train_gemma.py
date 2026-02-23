import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

print("Initialising GPU loading 4-bit gemma3")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it-bnb-4bit",
    max_seq_length=256,
    load_in_4bit=True,
    dtype=None,
)

print("Attaching LoRA")
#train small 16 adapter circuit
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=8,
    use_gradient_checkpointing="unsloth", #memory optimization
    random_state=3407,
)

print("loading dataset")
dataset = load_dataset("json", data_files="med_training_data.json", split="train")

#apply strict gemma3 protocol format
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

def formatting_prompt_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = f"<start_of_turn>user\n{instruction}\n\n{input_text}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(formatting_prompt_func, batched=True) #apply formatting

#training phase
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=256,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="paged_adamw_8bit",
        output_dir="outputs",
        seed=3407
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n"
)

#execute
trainer_stats = trainer.train()

print("Training complete")

#package lora adapter to gguf for local ollama inference
model.save_pretrained_gguf("gemma3-custom_med", tokenizer, quantization_method="q4_k_m")

print("Hardware cycle finished")