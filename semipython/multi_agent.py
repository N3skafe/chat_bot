# multi_agent.py
from typing import Dict, List, Any, Optional, Literal, TypedDict
from langgraph.graph import StateGraph

from ollama_client import MultiModelSystem, OllamaModel
from vector_store import VectorStore
from rag import RAG

# Define the state
class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    current_model: str
    final_answer: Optional[str]

class MultiAgentSystem:
    def __init__(self, models: MultiModelSystem, vector_store: VectorStore):
        self.models = models
        self.vector_store = vector_store
        
        # Create RAG systems for each model
        self.rag_systems = {
            model_name: RAG(vector_store, model) 
            for model_name, model in models.models.items()
        }
        
        # Build the agent workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the multi-agent workflow using LangGraph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("retriever", self._retrieve_documents)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("deep_research", self._deep_research)
        workflow.add_node("summarizer", self._summarize)
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._decide_if_skip_retrieval,
            {
                "retrieve": "retriever",
                "direct": "rag_agent"
            }
        )
        
        # Add other edges
        workflow.add_edge("retriever", "rag_agent")
        workflow.add_edge("rag_agent", "deep_research")
        workflow.add_edge("deep_research", "summarizer")
        
        # Set entry and exit points
        workflow.set_entry_point("router")
        workflow.set_finish_point("summarizer")
        
        return workflow.compile()
    
    def _route_query(self, state: AgentState) -> AgentState:
        """Route the query to the appropriate model based on content"""
        query = state["query"]
        
        # Simple routing logic based on query content
        if any(word in query.lower() for word in ["code", "programming", "develop", "function"]):
            state["current_model"] = "deepseek"  # deepseek for code
        elif any(word in query.lower() for word in ["reasoning", "complex", "analyze", "philosophy"]):
            state["current_model"] = "llama3"    # llama3 for complex reasoning
        else:
            state["current_model"] = "gemma"     # gemma for general queries
            
        return state
    
    def _decide_if_skip_retrieval(self, state: AgentState) -> str:
        """Decide whether to skip retrieval based on the query"""
        query = state["query"]
        
        # Skip retrieval for certain types of queries
        if any(x in query.lower() for x in ["hello", "hi", "greet", "who are you"]):
            return "direct"
        else:
            return "retrieve"
    
    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents from the vector store"""
        query = state["query"]
        documents = self.vector_store.search(query, n_results=5)
        state["documents"] = documents
        return state
    
    def _rag_agent(self, state: AgentState) -> AgentState:
        """Use RAG to generate an initial response"""
        query = state["query"]
        model_name = state["current_model"]
        
        # Check if we have documents or should answer directly
        if state.get("documents"):
            # Use RAG for answering
            response = self.rag_systems[model_name].query(query)
        else:
            # Direct question answering without documents
            prompt = f"Please answer the following question directly:\n\n{query}"
            response = self.models.get_model(model_name).generate(prompt)
        
        # Update state with the initial answer
        state["initial_answer"] = response
        
        # Add to chat history
        if "chat_history" not in state:
            state["chat_history"] = []
            
        state["chat_history"].append({"role": "user", "content": query})
        state["chat_history"].append({"role": "assistant", "content": response})
        
        return state
    
    def _deep_research(self, state: AgentState) -> AgentState:
        """Perform deeper research on complex queries"""
        query = state["query"]
        initial_answer = state.get("initial_answer", "")
        
        # Use llama3 model for deeper research regardless of initial model
        model = self.models.get_model("llama3")
        
        research_prompt = f"""
You have been given an initial answer to the query: "{query}"

Initial answer: "{initial_answer}"

Please research this topic more deeply and provide additional insights, facts, or context
that would make the answer more comprehensive. If you find any potential inaccuracies
in the initial answer, please correct them.
"""
        
        deeper_insights = model.generate(research_prompt)
        state["deeper_insights"] = deeper_insights
        
        return state
    
    def _summarize(self, state: AgentState) -> AgentState:
        """Summarize and finalize the answer"""
        query = state["query"]
        initial_answer = state.get("initial_answer", "")
        deeper_insights = state.get("deeper_insights", "")
        
        # Use the deepseek model for summarization
        model = self.models.get_model("deepseek")
        
        summarization_prompt = f"""
Original query: "{query}"

Initial response: "{initial_answer}"

Additional research: "{deeper_insights}"

Please synthesize a final, comprehensive answer that combines the initial response with the
additional research. The answer should be well-structured, accurate, and directly address
the original query.
"""
        
        final_answer = model.generate(summarization_prompt)
        state["final_answer"] = final_answer
        
        # Update chat history
        state["chat_history"].append({"role": "assistant", "content": final_answer})
        
        return state
    
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Process a user query and generate a response"""
        # Initialize state
        state = {
            "query": query,
            "chat_history": chat_history or [],
            "documents": [],
            "current_model": "gemma",  # Default model
            "final_answer": None
        }
        
        # Run the workflow
        result = self.workflow.invoke(state)
        
        # Return the final answer
        return result["final_answer"]
