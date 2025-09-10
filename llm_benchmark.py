import time
import requests
import json
from typing import Dict, Any

class LLMBenchmark:
    """Simple class to benchmark LLM performance with different query complexities."""
    
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1/completions"):
        self.lm_studio_url = lm_studio_url
        
    def call_llm(self, prompt: str) -> tuple[str, float]:
        """Call the LLM and return response + timing."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 0.9,
            "stream": False
        }
        
        start_time = time.time()
        try:
            response = requests.post(self.lm_studio_url, headers=headers, json=payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result["choices"][0]["message"]["content"]
                return llm_response, end_time - start_time
            else:
                return f"Error: {response.status_code}", end_time - start_time
                
        except Exception as e:
            end_time = time.time()
            return f"Exception: {str(e)}", end_time - start_time
    
    def run_benchmark(self):
        """Run benchmark tests with simple, medium, and complex queries."""
        
        # Test queries of increasing complexity
        test_queries = {
            "Simple": "What is 2+2?",
            "Medium": "Generate a SQL query to find all customers who made orders in the last 30 days, including their total order value.",
            "Complex": "You are a Sales & CRM agent. Analyze this user query and determine the action needed. Query: 'show me all cancelled orders from high-value customers in the last quarter, sorted by order value'. You handle: customers, leads, orders, support tickets, products. Classify the query into one of these action types: new_lead, new_order, support_ticket, retrieve_customers, retrieve_orders, retrieve_leads, retrieve_products, other. Also determine any specific filters or conditions mentioned. Respond with ONLY a JSON object with action_type, filters, and context fields."
        }
        
        print("🚀 LLM Performance Benchmark")
        print("=" * 50)
        
        results = {}
        
        for complexity, query in test_queries.items():
            print(f"\n📊 Testing {complexity} Query...")
            print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            response, duration = self.call_llm(query)
            
            results[complexity] = {
                "duration": duration,
                "response_length": len(response),
                "response_preview": response[:200] + "..." if len(response) > 200 else response
            }
            
            print(f"⏱️  Duration: {duration:.2f} seconds")
            print(f"📝 Response Length: {len(response)} characters")
            print(f"💬 Response Preview: {response[:100]}{'...' if len(response) > 100 else ''}")
            print("-" * 30)
        
        # Summary
        print("\n📈 BENCHMARK SUMMARY")
        print("=" * 50)
        for complexity, result in results.items():
            print(f"{complexity:8}: {result['duration']:.2f}s | {result['response_length']} chars")
        
        # Performance analysis
        simple_time = results["Simple"]["duration"]
        medium_time = results["Medium"]["duration"] 
        complex_time = results["Complex"]["duration"]
        
        print(f"\n🔍 Performance Analysis:")
        print(f"Medium vs Simple: {medium_time/simple_time:.1f}x slower")
        print(f"Complex vs Simple: {complex_time/simple_time:.1f}x slower")
        print(f"Complex vs Medium: {complex_time/medium_time:.1f}x slower")
        
        return results

if __name__ == "__main__":
    benchmark = LLMBenchmark()
    benchmark.run_benchmark()
