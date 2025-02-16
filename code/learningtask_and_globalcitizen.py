import random
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
import random
import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging
from huggingface_hub import login
from datasets import load_dataset


@dataclass
class Trial:
    trial_id: int
    trial_type: str
    statement: str
    avail_sources: List[str] 
    participant_response: bool
    source_name: str
    source_response: bool
    confidence: float
    final_response: bool = None
    final_confidence: float = None
    correct_answer: bool = None
    timestamp: str = None

    def to_dict(self):
        return {
            "trial_id": self.trial_id,
            "trial_type": self.trial_type,
            "statement": self.statement,
            "avail_sources": self.avail_sources,
            "participant_response": self.participant_response,
            "source_name": self.source_name,
            "source_response": self.source_response,
            "confidence": self.confidence,
            "final_response": self.final_response,
            "final_confidence": self.final_confidence,
            "correct_answer": self.correct_answer,
            "timestamp": self.timestamp
        }

class ParticipantAgent:
    def __init__(self, name: str, model, tokenizer):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.source_memory = defaultdict(lambda: {"agreements": 0, "total": 0, "correct": 0})
        self.trials = []
        self.current_trial_id = 0
    
    def update_memory(self, source_name: str, agreed: bool, correct: bool = None):
        memory = self.source_memory[source_name]
        memory["total"] += 1
        if agreed:
            memory["agreements"] += 1
        if correct is not None:
            memory["correct"] += 1 if correct else 0
    
    def get_source_stats(self) -> Dict:
        stats = {}
        for source_name, memory in self.source_memory.items():
            total = memory["total"]
            stats[source_name] = {
                "agreement_rate": memory["agreements"] / total if total > 0 else 0,
                "accuracy_rate": memory["correct"] / total if total > 0 else 0
            }
        return stats
    
    def generate_response(self, prompt: str, response_type: str = 'binary', stats: Dict = None) -> any:
        memory_context = ""
        if stats:
            memory_context = "\nSource information:\n" + "\n".join([
                f"- {name}: Agreement rate: {s['agreement_rate']:.1%}, "
                f"Accuracy rate: {s['accuracy_rate']:.1%}"
                for name, s in stats.items()
            ])
        
        formatted_prompt = (
            f"System: Assume you are {self.name}. \n"
            f"You scored the following on a survey (each question in the range 0-5, 5 being highly agree and 0 being highly disagree): \n"
            f"1. Most people who are important to me think that being a global citizen is desirable: 5. \n" 
            f"2. If I called myself a global citizen most people who are important to me would approve: 5   \n"
            f"3. I understand how the various cultures of this world interact socially: 5\n"
            f"4. I am aware that my actions in my local environment may affect people in other countries: 5 \n"
            f"5. I try to stay informed of current issues that impact international relations: 5 \n"
            f"6. I believe that Iam connected to people in other countries, and my actions can affect them: 5 \n"
            f"7. I would describe myself as a global citizen: 5\n"
            f"8. I strongly identify with global citizens: 5\n"
            f"9. I am able to empathize with people from other countries: 5 \n"
            f"10. It is easy for me to put myself in someone else’s shoes regardless of what country they are from: 5\n"
            f"11. I would like to join groups that emphasize getting to know people from different countries: 5\n"
            f"12. I am interested in learning about the many cultures that have existed in this world: 5\n"
            f"13. Those countries that are well off should help people in countries who are less fortunate: 5\n"
            f"14. Basic services such as healthcare, clean water, food, and legal assistance should be available to everyone, regardless of what country they live in: 5\n"
            f"15. People have a responsibility to conserve natural resources to foster a sustainable environment: 5\n"
            f"16. Natural resources should be used primarily to provide for basic needs rather than material wealth: 5\n"
            f"17. ‘If I had the opportunity, I would help others who are in need regardless of their nationality: 5\n"
            f"18. If I could, I would dedicate my life to helping others no matter what country they are from: 5\n"
            f"19. Being actively involved in global issues is my responsibility: 5\n"
            f"It is my responsibility to understand and respect cultural differences across the globe to the best of my abilities: 5\n"
            f"Respond concisely based on who you are.{memory_context}\n\n"
            f"Human: {prompt}\n\n"
            f"Assistant: "
        )
        
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            temperature=0.2,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
        
        if response_type == 'binary':
            return 'yes' in response
        elif response_type == 'confidence':
            try:
                confidence = float(''.join(filter(str.isdigit, response)))
                return min(max(confidence, 0), 100)
            except:
                return 50.0
        elif response_type == 'source_choice':
            return '1' in response
        
        return response

def get_source_response(source_name: str, participant_response: bool, correct_answer: bool = None) -> bool:

    if source_name == "m1":
        return correct_answer if correct_answer is not None else participant_response
    elif source_name == "m2":
        return correct_answer if correct_answer is not None else not participant_response
    elif source_name == "m3":
        return participant_response
    else:  # Random_Disagree
        return not participant_response

def run_experiment(participant: ParticipantAgent):
    # example stamements, we provide more choices for each
    political_statements = [
        "Universal basic income would reduce inequality",
        "Climate change requires immediate government action",
        "Healthcare should be a fundamental right",
        "Free market capitalism is the best economic system",
        "Immigration enriches our culture and economy"
    ]
    
    merlin_trials = [
        ("The black cat sits near the window", True),
        ("A quick dog jumps over fence", False),
        ("Two birds fly in the garden", True),
        ("The loud car moves street", False),
        ("A bright star shines above mountain", True),
        ("The gentle breeze sways the trees", True),   
        ("Small fish swim through the water", False),      
        ("The white clouds float across the sky", True), 
        ("Old book lies on the table", False),             
        ("The yellow butterfly lands on the flower", True) 
    ]
    
    
    # sources = ["AlwaysRight_Disagree","Random_Disagree", "AlwaysRight_Agree", "Random_Agree"]
    
    # sources1 = ["AlwaysRight_Agree", "Random_Agree"]
    # sources2 = ["AlwaysRight_Disagree","Random_Disagree"]


    trial_id = 0
    
    # Learning Stage
    print("Starting Learning Stage...")
    for _ in range(10):
        trial_id += 1
        statement, is_merlin = random.choice(merlin_trials)
        source = random.choice(sources)

        # merlin pattern
        prompt = f"Is this a Merlin pattern? '{statement}' Answer yes/no."
        participant_response = participant.generate_response(prompt)
        source_response = get_source_response(source, participant_response)
        
        participant.update_memory(
            source,
            agreed=None,
            correct=(participant_response == is_merlin)
        )
        
        participant.update_memory(
            source,
            agreed=None,
            correct=(source_response == is_merlin)
        )
        
        # political statement
        
        statement = random.choice(political_statements)
        
        prompt = f"Do you agree? '{statement}' Answer yes/no."
        participant_response = participant.generate_response(prompt)
        source_response = get_source_response(source, participant_response)
        
        participant.update_memory(
            source,
            agreed=(participant_response == source_response)
        )
        
        trial = Trial(
            trial_id=trial_id,
            trial_type="political",
            statement=statement,
            avail_sources= [source],
            participant_response=participant_response,
            source_name=source,
            source_response=source_response,
            confidence=100,
            timestamp=datetime.now().isoformat()
        )
        participant.trials.append(trial)
    
        
        
    print("Starting Choice Stage...")
    for statement, is_merlin in merlin_trials:
        trial_id += 1
        stats = participant.get_source_stats()
        
        # Initial response
        prompt = f"Is this a Merlin pattern? '{statement}' Answer yes/no."
        initial_response = participant.generate_response(prompt, stats=stats)
        initial_confidence = participant.generate_response(
            "Rate your confidence (1-100):",
            response_type='confidence',
            stats=stats
        )
        
        # Source selection
        available_sources1 = random.sample(sources1, 1)
        available_sources2 = random.sample(sources2, 1)
        available_sources = available_sources1 + available_sources2
        source_choice = participant.generate_response(
            f"Choose source (1) {available_sources[0]} or (2) {available_sources[1]} to look at their answer:",
            response_type='source_choice',
            stats=stats
        )
        # Remember you have to choose a source that has a better shot at being correct in the merlin task.
        chosen_source = available_sources[0] if source_choice else available_sources[1]
        source_response = get_source_response(chosen_source, initial_response, is_merlin)
        
        # Update memory for all available sources
        for source in available_sources:
            source_resp = get_source_response(source, initial_response, is_merlin)
            participant.update_memory(
                source,
                agreed=(initial_response == source_resp),
                correct=(source_resp == is_merlin)
            )
        
        # Final response
        final_response = participant.generate_response(
            f"Would you like to change your answer based on {stats}? Current: {'yes' if initial_response else 'no'}",
            stats=stats
        )
        
        # Update memory again with final response
        participant.update_memory(
            chosen_source,
            agreed=(final_response == source_response),
            correct=(source_response == is_merlin)
        )
        
        final_confidence = participant.generate_response(
            "Rate your final confidence (1-100):",
            response_type='confidence',
            stats=stats
        )
        
        
        trial = Trial(
            trial_id=trial_id,
            trial_type="merlin",
            statement=statement,
            avail_sources= available_sources,
            participant_response=initial_response,
            source_name=chosen_source,
            source_response=source_response,
            confidence=initial_confidence,
            final_response=final_response,
            final_confidence=final_confidence,
            correct_answer=is_merlin,
            timestamp=datetime.now().isoformat()
        )
        participant.trials.append(trial)
    
    # Save results
    results = {
        "participant_name": participant.name,
        "trials": [trial.to_dict() for trial in participant.trials],
        "source_memory": participant.source_memory
    }
    
    with open(f"experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2)

def main():
    
    # Setup logging and model
    
    logging.set_verbosity_error()
    login(LOGIN_ID)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    print("Loading model and tokenizer...")
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    # model_name = "Qwen/Qwen2.5-72B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map='auto',
        torch_dtype=torch.float16
    )
    
    participant = ParticipantAgent("Participant1", model, tokenizer)
    run_experiment(participant)
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()
