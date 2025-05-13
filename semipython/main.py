# main.py
#!/usr/bin/env python3
import argparse
from app import demo

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent RAG System with Ollama Models")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the Gradio app on")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting Multi-Agent RAG System at http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
