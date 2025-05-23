import json
from typing import Dict, List, Tuple, Annotated, Optional
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from openai import OpenAI
import textwrap
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
import pdfplumber
from datetime import datetime

# Load env
load_dotenv(override=True)  

env_path = os.path.join(os.getcwd(), '.env')
# print(f".env file exists: {os.path.exists(env_path)}")

if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        env_contents = f.read().strip()
        print(f"API key format check: {env_contents.startswith('OPENAI_API_KEY=sk-')}")

# Text extraction from pdf as input 
def extract_text_from_pdf(pdf_path: str) -> str:
    text = "" 
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() 
            if page_text: 
                text += page_text + "\n"
    return text.strip()  

# Define the state that will be passed between nodes
class AgentState(BaseModel):
    document_text: str
    current_prompt: str
    previous_outputs: List[Dict] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3
    improvement_threshold: float = 0.8
    current_quality_score: float = 0.0
    
    # define quality metrics
    current_quality_metrics: Dict[str, float] = Field(default_factory=lambda: {
        "completeness": 0.0,
        "accuracy": 0.0,
        "clarity": 0.0,
        "technical_depth": 0.0
    })
    # store scores in list
    dimension_history: Dict[str, List[float]] = Field(default_factory=lambda: {
        "completeness": [],
        "accuracy": [],
        "clarity": [],
        "technical_depth": []
    })
    quality_analysis: Optional[str] = None
    final_output: str = ""
    system_prompt: str = ""
    quality_progression: Optional[Dict] = None

# calculate weighted average of quality metrics to generate overall score
class QualityMetrics(BaseModel):
    completeness: float
    accuracy: float
    clarity: float
    technical_depth: float
    
    def overall_score(self) -> float:
        weights = {
            "completeness": 0.3,
            "accuracy": 0.3,
            "clarity": 0.2,
            "technical_depth": 0.2
        }
        return sum(getattr(self, metric) * weight 
                  for metric, weight in weights.items())

# agent class structure
class RecursiveAgent:
    def __init__(self, testing: bool = False):
        self.testing = testing
        
        if testing:
            self.client = self._mock_client()
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables. Please set OPENAI_API_KEY in your .env file.")
            self.client = OpenAI(api_key=api_key)
        self.prompt_sets = self._load_prompt_sets()
        
    # mock client for testing 
    def _mock_client(self):
        class MockCompletions:
            def create(self, **kwargs):
                return type("MockResponse", (), {
                    "choices": [type("Choice", (), {
                        "message": type("Message", (), {"content": "[MOCKED OUTPUT: input echoed back]\n" 
                                                        + kwargs["messages"][-1]["content"]})
                    })()]
                })()
        class MockChat:
            completions = MockCompletions()
        class MockClient:
            chat = MockChat()
        return MockClient()

        
    def _load_prompt_sets(self) -> Dict:
        with open('prompt_sets.json', 'r') as f:
            return json.load(f)
    
    def assess_quality(self, state: AgentState) -> AgentState:
        
        # Quality prompt that requests scores for each dimension
        quality_prompt = f"""
        Assess the quality of the following output based on these criteria:
        1. Completeness (0-1): How thoroughly does the output address all aspects of the task?
        2. Accuracy (0-1): How factually correct and reliable is the information provided?
        3. Clarity (0-1): How clear, well-organized, and easy to understand is the output?
        4. Technical depth (0-1): How well does the output demonstrate appropriate expertise and depth?
        
        Output to assess:
        {state.final_output}
        
        Previous outputs for context:
        {json.dumps(state.previous_outputs, indent=2)}
        
        Analyze each criterion individually and provide a detailed assessment.
        Then provide scores for EACH criterion using exactly this format at the end of your analysis:
        
        COMPLETENESS: 0.XX
        ACCURACY: 0.XX
        CLARITY: 0.XX
        TECHNICAL_DEPTH: 0.XX
        
        Each score must be a number between 0 and 1.
        """
        
        # Request evaluation from gpt-4
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a quality assessment expert. Always evaluate each criterion separately and provide individual scores."},
                {"role": "user", "content": quality_prompt}
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract and parse individual dimension scores from response
        metrics = {
            "completeness": 0.0,
            "accuracy": 0.0,
            "clarity": 0.0,
            "technical_depth": 0.0
        }
  
        for line in content.split('\n'):
            for dimension in metrics.keys():
                prefix = f"{dimension.upper()}:"
                if line.strip().startswith(prefix):
                    try:
                        score_text = line.strip()[len(prefix):].strip()
                        metrics[dimension] = float(score_text)
                    except ValueError:
                        pass # if failed, default score
        
        # qualityMetrics instance
        quality_metrics = QualityMetrics(
            completeness=metrics["completeness"],
            accuracy=metrics["accuracy"],
            clarity=metrics["clarity"],
            technical_depth=metrics["technical_depth"]
        )
        
        # calculate overall score
        overall_score = quality_metrics.overall_score()
        
        # Update state with quality metrics
        state.current_quality_metrics = metrics
        state.current_quality_score = overall_score
        
        # Update dimension history
        for dimension, score in metrics.items():
            state.dimension_history[dimension].append(score)
        
        # Add analysis to context for more targeted improvements
        state.quality_analysis = content
        
        return state

    # Generate improved version of output
    def generate_improvement(self, state: AgentState) -> AgentState:
        
        # Find the weakest dimensions
        weakest_dimensions = sorted(
            state.current_quality_metrics.items(), 
            key=lambda x: x[1]
        )[:2]  # Focus on the two lowest scoring dimensions
        
        dimension_guidance = "\n".join([
            f"- {dimension.capitalize()} (current score: {score:.2f}): " +
            self._get_dimension_guidance(dimension)
            for dimension, score in weakest_dimensions
        ])
        
        improvement_prompt = f"""
        Based on the current output and quality assessment, you need to create an improved version.
        
        Current output:
        {state.final_output}
        
        Quality assessment:
        {state.quality_analysis}
        
        Focus especially on improving these dimensions:
        {dimension_guidance}
        
        Previous outputs:
        {json.dumps(state.previous_outputs, indent=2)}
        
        Generate a new, improved version of the output that maintains the strengths while addressing the weaknesses.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": state.system_prompt},
                {"role": "user", "content": improvement_prompt}
            ],
            temperature=0.1
        )
        
        improved_output = response.choices[0].message.content.strip()
        state.final_output = improved_output
        state.previous_outputs.append({
            "iteration": state.iteration_count,
            "output": improved_output,
            "quality_score": state.current_quality_score,
            "quality_metrics": state.current_quality_metrics
        })
        state.iteration_count += 1
        return state

    def _get_dimension_guidance(self, dimension: str) -> str:
       
        guidance = {
            "completeness": "Ensure all aspects of the task are thoroughly addressed. Check for missing information or unexplored angles.",
            "accuracy": "Verify factual statements and ensure all information is correct and properly contextualized.",
            "clarity": "Improve organization, use clearer language, and ensure logical flow between ideas.",
            "technical_depth": "Add more specialized knowledge, relevant details, and demonstrate deeper understanding of the subject matter."
        }
        return guidance.get(dimension, "Focus on overall improvement.")

    # end iteration logic
    def should_continue(self, state: AgentState) -> str:
       
        # check max iterations
        if state.iteration_count >= state.max_iterations:
            return "complete"
        
        # check overall quality threshold
        if state.current_quality_score >= state.improvement_threshold:
            return "complete"
        
        # Check for diminishing returns
        if state.iteration_count >= 1:
            previous_score = state.previous_outputs[-1]["quality_score"]
            improvement = state.current_quality_score - previous_score
            
            # < threshold, stop
            if improvement < 0.05 and state.iteration_count >= 2:
                return "complete"
            
            # neg progress, stop
            if improvement < 0:
                return "complete"
        
        # check min thresholds for individual dimensions
        critical_dimensions = ["completeness", "accuracy"]
        min_critical_score = 0.7
        
        if all(state.current_quality_metrics[dim] >= min_critical_score for dim in critical_dimensions):
            # if all dimensions meet thresholds, accept
            if state.current_quality_score >= state.improvement_threshold * 0.9:
                return "complete"
        
        return "continue"

    # Final node that returns the state unchanged
    def end_workflow(self, state: AgentState) -> AgentState:
        return state

    def create_recursive_workflow(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("assess_quality", self.assess_quality)
        workflow.add_node("generate_improvement", self.generate_improvement)
        workflow.add_node("end", self.end_workflow)
        
        # Set entry point
        workflow.set_entry_point("assess_quality")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "assess_quality",
            self.should_continue,
            {
                "continue": "generate_improvement",
                "complete": "end"
            }
        )
        
        # Add edge from generate_improvement back to assess_quality
        workflow.add_edge("generate_improvement", "assess_quality")
        
        return workflow.compile()
    
    # # Visualize quality progression across iterations
    # def visualize_quality_progression(self, state: AgentState) -> Dict:
        
    #     iterations = list(range(state.iteration_count + 1))
        
    #     # Prepare dimension history
    #     dimension_data = {
    #         dimension: scores
    #         for dimension, scores in state.dimension_history.items()
    #     }
        
    #     # Prepare overall scores
    #     overall_scores = [output.get("quality_score", 0) for output in state.previous_outputs]
    #     if state.current_quality_score and len(overall_scores) < len(iterations):
    #         overall_scores.append(state.current_quality_score)
        
    #     return {
    #         "iterations": iterations,
    #         "dimensions": dimension_data,
    #         "overall_scores": overall_scores
    #     }

    def process_prompt(self, document_text: str, prompt: str, system_prompt: str) -> AgentState:
        initial_state = AgentState(
            document_text=document_text,
            current_prompt=prompt,
            system_prompt=system_prompt,
            max_iterations=3,
            improvement_threshold=0.8
        )
        
        # init generation
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        initial_state.final_output = response.choices[0].message.content.strip()
        
        # run workflow
        workflow = self.create_recursive_workflow()
        final_state = workflow.invoke(initial_state)
        
        # Ensure final_state is an AgentState object
        if not isinstance(final_state, AgentState):
            final_state = AgentState(**final_state)
        
        # # Add quality progression data
        # final_state.quality_progression = self.visualize_quality_progression(final_state)
        
        return final_state

    def process_prompt_set(self, document_text: str, set_name: str) -> Dict:
        if set_name not in self.prompt_sets:
            raise ValueError(f"Prompt set '{set_name}' not found")
            
        config = self.prompt_sets[set_name]
        system_prompt = config["system_prompt"]
        prompts = config["prompts"]
        
        results = {}
        for key, instruction in prompts.items():
            if key != "sub-prompts":
                full_prompt = f"""
                Document:
                {document_text}
                
                Task:
                {instruction}
                """
                
                # DEBUG - content type
                if key == "content_type":
                    print(f"DEBUG - CONTENT_TYPE prompt being sent:")
                    print(f"Document length: {len(document_text)}")
                    print(f"Instruction: {instruction}")
                    print(f"Full prompt first 500 chars: {full_prompt[:500]}")
                    print("-" * 50)
                
                final_state = self.process_prompt(document_text, full_prompt, system_prompt)
        
                state_dict = final_state.model_dump()
                
                results[key] = {
                    "output": state_dict.get("final_output", "No output generated"),
                    "quality_score": state_dict.get("current_quality_score", 0.0),
                    "iterations": state_dict.get("iteration_count", 0),
                    "history": state_dict.get("previous_outputs", [])
                }
        
        return results

    def process_all_prompt_sets(self, document_text: str) -> Dict:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_prompt_set, document_text, set_name)
                for set_name in self.prompt_sets
            ]
            results = [future.result() for future in futures]
        return dict(zip(self.prompt_sets.keys(), results))
    
    

def main():

	# pdf extraction 
    pdf_path = "test_1page.pdf"
    document_text = extract_text_from_pdf(pdf_path=pdf_path)
    # print(f"PDF extracted successfully: {len(document_text)} characters")
    # print(f"First 200 chars: {document_text[:200]}")
    
	# init agent 
    agent = RecursiveAgent(testing=False)
    

    # short text test input
    short_document_text = """
    Dynamic Mission Replanning System - Project Plan
    The Dynamic Mission Replanning System aims to develop a sophisticated software
    framework that enables real-time adaptation of mission parameters for future fighter jets
    based on changing operational conditions.
    """
    
    # Process a single prompt set
    results = agent.process_prompt_set(document_text, "metadata")
    #results = agent.process_prompt_set(short_document_text, "metadata")
    
    # print(json.dumps(results, indent=2))
    
    # Process all prompt sets
    # all_results = agent.process_all_prompt_sets(document_text)
    # print(json.dumps(all_results, indent=2))
    
    # store in txt file
    output_filename = f"results_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("METADATA ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        for task_name, result in results.items():
            f.write(f"TASK: {task_name.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Quality Score: {result['quality_score']:.3f}\n")
            f.write(f"Iterations: {result['iterations']}\n")
            f.write(f"Final Output:\n")
            f.write("-" * 20 + "\n")
            
            # Process the output to handle \n characters properly
            output_text = result['output']
            if isinstance(output_text, str):
                # Replace literal \n with actual line breaks
                formatted_output = output_text.replace('\\n', '\n')
                f.write(formatted_output)
            else:
                f.write(str(output_text))
            
            f.write("\n\n")
            
            # Add iteration history if available
            if result.get('history'):
                f.write("ITERATION HISTORY:\n")
                f.write("-" * 20 + "\n")
                for i, hist in enumerate(result['history']):
                    f.write(f"Iteration {i+1}: Score {hist.get('quality_score', 0):.3f}\n")
                f.write("\n")
            
            f.write("=" * 60 + "\n\n")
    
    print(f"Results saved to {output_filename}")
    

if __name__ == "__main__":
    main() 