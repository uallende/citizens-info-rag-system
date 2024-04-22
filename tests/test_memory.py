import os, torch
from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
from app.rag import generate_text_embeddings

def main():

    max_length = 4096
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')

    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    os.chdir(parent_dir)

    load_dotenv()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModel.from_pretrained(
        'Salesforce/SFR-Embedding-Mistral',
        trust_remote_code=True,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    input_text = f'What is the best way to bring my stuff and lugages to Ireland'
    embeddings = generate_text_embeddings(model=model,
                                          tokenizer=tokenizer,
                                          text=input_text,
                                          max_length=max_length
                                          )

    print(embeddings)

if __name__ == "__main__":
    main()