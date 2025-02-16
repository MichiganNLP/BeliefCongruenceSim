import random
import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging
from huggingface_hub import login
from datasets import load_dataset


# warnings
logging.set_verbosity_error()

login(LOGIN_ID)

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Model and tokenizer setup
model_name = "Qwen/Qwen2.5-72B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    device_map='auto',
    torch_dtype=torch.float16
)

model = torch.compile(model)

def clean_text(text):
    # Remove triple quotes at start and end
    text = text.strip('"""')
    
    # Remove all @ symbols and sequences
    text = text.replace("@", "")
    
    # Remove other special characters and quotes
    text = text.replace("``", "")
    text = text.replace("''", "")
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

def data():
    ds = load_dataset("newsmediabias/fake_news_elections_labelled_data",  use_auth_token=True)
    df = ds['train'].to_pandas()

    # See basic information about the dataset
    print(df.info())
    print("\nSample of the data:")
    print(df.head())
    
    clean = []
    
    for i in df.text.values:
        clean.append(clean_text(i))
        
    return clean
    

class Agent:
    def __init__(self, name, political, can_create_agents, can_halt_agents, plugins):
        self.conversation_history = []
        # self.race = race 
        self.name = name
        self.political = political
        self.can_create_agents = can_create_agents
        self.can_halt_agents = can_halt_agents
        self.plugins = plugins 
        self.state = None
        self.memory_lst = []
        
    def generate_response(self, prompt, response_type='discussion'):
        constrained_prompts = {
            'discussion': (
                f"{prompt}\n"
                "RESPONSE INSTRUCTIONS:\n"
                "1. Format: '[AGENT NAME]: <specific response>'\n"
                "2. Start with your specific agent name\n"
                "3. Provide a clear, concise response.\n"
                "4. Limit to 3 sentences maximum.\n\n"
                "example format: [AGENT1]: Misinformation because the claim does not make sense.\n"
                "[AGENT2]: True information because the claim is statistically proven.\n"
                "{self.name}:     \n"
            ),
        }
        
        
        input_ids = tokenizer(constrained_prompts[response_type], return_tensors="pt").input_ids.to(model.device)
        
        # Generate response with carefully tuned parameters
        outputs = model.generate(
            input_ids, 
            max_length=input_ids.shape[1] + 100,  # Reduced response length
            num_return_sequences=1, 
            temperature=0.5,  
            top_p=0.9,        
            do_sample=True,
            repetition_penalty=1.5,  # Increased to reduce repetition
            eos_token_id=tokenizer.eos_token_id
        )
            
        # # Decode the response
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if response_type == 'discussion':
            response_text = raw_response.split("{self.name}:")[-1].strip() 
            return response_text

    def create_agent(self, role, can_create_agents, can_halt_agents, plugins):
        if self.can_create_agents:
            return Agent(role, can_create_agents, can_halt_agents, plugins)
        else:
            raise Exception("This agent does not have the ability to create new agents")

    def halt_agent(self, role, can_create_agents, can_halt_agents, plugins):
        if self.can_halt_agents:
            return Agent(role, can_create_agents, can_halt_agents, plugins)
        else:
            raise Exception("This agent does not have the ability to halt agents")

    def add_event(self, event):
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory):
        self.memory_lst.append({"role": "assistant", "content": memory})
        
        
def run_experiments():
    responses__ = []
    intial_resp = []
    final_resp = []

    news = data()
    
    # Run multiple rounds of experiments
    cnt = 0
    for i in news:
        
            print(i)
            print("Conv starts....")
            m1 = Agent("m1", "republican", True, True, ["Language generation"])
            m2 = Agent("m2", "republican", True, True, ["Language generation"])
            m3 = Agent("m3", "democrat", True, True, ["Language generation"])
            m4 = Agent("m4", "democrat", True, True, ["Language generation"])

            agents = [m1, m2, m3, m4]

            # Discussion rounds
            for agent in agents:
                prompt = f"Assume you are {agent.name}. \
                Given the claim: {i}. \
                Based on your background as a {agent.political} person, determine if this is true information or misinformation."
                resp = agent.generate_response(prompt, response_type='discussion')
                print(f"{agent.name}: {resp}\n")
                responses__.append(f"{agent.name} Agent: {resp}\n")
                agent.add_memory(resp)
                intial_resp.append(f"{agent.name} Agent: {resp}\n")

            # accuracy nudge
            for agent in agents:
                prompt = f"Assume you are {agent.name}. \
                Given the claim: {i}. \
                Based on your background as a {agent.political} person, determine if this is true information or \
                misinformation. Remember, your task is to accurately determine whether the information is true or false. \
                Avoid letting your political affiliation influence your judgment."
                resp = agent.generate_response(prompt, response_type='discussion')
                print(f"{agent.name}: {resp}\n")
                responses__.append(f"{agent.name} Agent: {resp}\n")
                agent.add_memory(resp)
                    
            for agent in agents:
                for round_num in range(1):
                    prompt = f"Assume you are {agent.name}. \
                    Given the claim: {i}. \
                    Based on {agent.memory_lst[0]['content']}, convince others about your own perspective. \
                    You have to choose one and respond in this format: <true information or misinformation> <brief reason>. Limit responses to 2 sentences. "
                    resp = agent.generate_response(prompt, response_type='discussion')
                    print(f"{agent.name}: {resp}\n")
                    responses__.append(f"{agent.name} Agent: {resp}\n")
                    agent.add_memory(resp)
                for round_num in range(1):                   
                    prompt = f"Assume you are {agent.name}. \
                    Given the claim: {i}. \
                    Based on {agent.memory_lst[0]['content']}, be open to other perspective so that you can come to a consensus. \
                    You have to choose one and respond in this format: <true information or misinformation> <brief reason>. Limit responses to 2 sentences. "
                    resp = agent.generate_response(prompt, response_type='discussion')
                    print(f"{agent.name}: {resp}\n")
                    responses__.append(f"{agent.name} Agent: {resp}\n")
                    agent.add_memory(resp)
            
            
            for agent in agents:
                prompt = f"Assume you are {agent.name}. \
                Given the claim: {i}. \
                Based on your background as a {agent.political} person and this additional context, refine your previous perspective and determine if this is true information or misinformation. Respond in this format: <true information or misinformation>"
                resp = agent.generate_response(prompt, response_type='discussion')
                print(f"{agent.name}: {resp}\n")
                responses__.append(f"{agent.name} Agent: {resp}\n")
                agent.add_memory(resp)
                final_resp.append(f"{agent.name} Agent: {resp}\n")
                
            cnt = cnt + 1
    # Save results
    pd.DataFrame({"responses": responses__}).to_csv("logs.csv")
    pd.DataFrame({"remembered": intial_resp}).to_csv("initialresp.csv")
    pd.DataFrame({"final_choice": final_resp}).to_csv("finalresp.csv")

# Run the experiments
if __name__ == "__main__":
    run_experiments()
    
