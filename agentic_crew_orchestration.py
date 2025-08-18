#configure base url in the code and key in secrets.json file
import os, json, httpx
import sys
from datetime import datetime
import time
import pandas as pd

os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, LLM

# Define the models to test
MODELS_TO_TEST = [
    # ✅ Working models (tested successfully)
    "gpt-4o",
    "gpt-4o-mini", 
    "bedrock-claude-4-sonnet",
    "bedrock-claude-3-5-sonnet-v2",
    "bedrock-claude-3-7-sonnet",
    "bedrock-claude-3-haiku",
    "gemini-2.0-flash",
    "gpt-4.1",
    "bedrock-claude-3-sonnet"
]

def setup_llm(model_name):
    """Set up CrewAI LLM with proper error handling for specific model"""
    try:
        # Load API key from secrets
        with open('secrets.json') as f:
            secrets = json.load(f)
        
        api_key = secrets['api_key']
        base_url = "put your base url here"
        
        # Set environment variables (CrewAI uses these automatically)
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_BASE_URL'] = base_url
        
        # Initialize CrewAI LLM with specific model
        llm = LLM(
            model=f'openai/{model_name}',
            api_key=api_key,
            api_base=base_url
        )
        
        print(f"✓ CrewAI LLM initialized successfully for model: {model_name}")
        return llm
        
    except FileNotFoundError:
        print("❌ Error: secrets.json file not found")
        return None
    except KeyError as e:
        print(f"❌ Error: Missing key in secrets.json: {e}")
        return None
    except Exception as e:
        print(f"❌ Error initializing CrewAI LLM for {model_name}: {e}")
        return None

# Define the predefined themes list
PREDEFINED_THEMES = [
    <list of golden themes here>
]

def create_agents(llm):
    """Create agents with the specified LLM"""
    # Agent 1: Theme Generator
    theme_generator_agent = Agent(
        role="Theme Classification Specialist",
        goal="Analyze text content and identify the most appropriate theme from predefined categories",
        backstory="""You are a specialized AI agent with expertise in natural language processing and content categorization. 
        Your primary responsibility is to analyze text content and classify it into one of the predefined themes. 
        You have extensive experience in understanding context, sentiment, and subject matter to make accurate classifications.
        You work collaboratively with a validation specialist to ensure the highest accuracy.""",
        llm=llm,
        verbose=True,
        allow_delegation=True
    )

    # Agent 2: Theme Validator
    theme_validator_agent = Agent(
        role="Theme Validation Expert",
        goal="Review and validate theme classifications to ensure accuracy and consistency",
        backstory="""You are a quality assurance specialist focused on validating theme classifications. 
        Your expertise lies in cross-referencing content with theme definitions, identifying potential misclassifications, 
        and ensuring consistency across similar content types. You work closely with the Theme Classification Specialist 
        to provide feedback and coordinate on challenging classifications. You have the authority to request 
        re-classification if the initial assessment doesn't meet quality standards.""",
        llm=llm,
        verbose=True,
        allow_delegation=True
    )
    
    return theme_generator_agent, theme_validator_agent

def create_classification_tasks(text_to_analyze, theme_generator_agent, theme_validator_agent):
    """Create collaborative tasks for theme classification and validation"""
    
    # Task 1: Initial Theme Classification
    theme_generation_task = Task(
        description=f"""Analyze the following text and classify it into ONE of these predefined themes:

{chr(10).join([f"- {theme}" for theme in PREDEFINED_THEMES])}

Text to analyze: "{text_to_analyze}"

Your task is to:
1. Carefully read and understand the text content
2. Consider the context, keywords, and overall meaning
3. Match the content to the most appropriate theme from the list
4. Provide your initial classification with a brief reasoning

Return your response in this format:
CLASSIFICATION: [theme name]
REASONING: [brief explanation of why this theme was selected]
CONFIDENCE: [High/Medium/Low]""",
        agent=theme_generator_agent,
        expected_output="A theme classification with reasoning and confidence level"
    )
    
    # Task 2: Theme Validation and Coordination
    theme_validation_task = Task(
        description=f"""Review the theme classification provided by the Theme Classification Specialist for the text: "{text_to_analyze}"

Your validation process should include:
1. Verify if the selected theme accurately represents the content
2. Check for any potential alternative themes that might be more suitable
3. Assess the reasoning provided by the classifier
4. Coordinate with the Theme Classification Specialist if changes are needed

Available themes for reference:
{chr(10).join([f"- {theme}" for theme in PREDEFINED_THEMES])}

Provide your validation in this format:
VALIDATION_STATUS: [APPROVED/NEEDS_REVISION]
FINAL_THEME: [confirmed theme name]
VALIDATOR_NOTES: [your assessment and any recommendations]
QUALITY_SCORE: [1-10 scale]""",
        agent=theme_validator_agent,
        expected_output="A validation assessment with final theme confirmation",
        context=[theme_generation_task]  # This task depends on the first task's output
    )
    
    return [theme_generation_task, theme_validation_task]

def run_collaborative_classification_for_model(model_name, text_samples):
    """Execute collaborative theme classification for a specific model with enhanced logging"""
    print(f"\n{'='*100}")
    print(f"🤖 TESTING MODEL: {model_name}")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total Samples to Process: {len(text_samples)}")
    print(f"{'='*100}")
    
    # Setup LLM for this model
    print(f"🔧 Initializing LLM for {model_name}...")
    llm = setup_llm(model_name)
    if llm is None:
        print(f"❌ Failed to initialize LLM for {model_name}")
        return {
            "model": model_name,
            "status": "failed",
            "error": "Failed to initialize LLM",
            "results": []
        }
    
    # Create agents for this model
    print(f"👥 Creating collaborative agents for {model_name}...")
    theme_generator_agent, theme_validator_agent = create_agents(llm)
    print(f"✅ Agents created successfully")
    
    results = []
    model_start_time = time.time()
    successful_count = 0
    failed_count = 0
    
    print(f"\n🚀 Starting processing of {len(text_samples)} samples...")
    
    for i, text in enumerate(text_samples, 1):
        print(f"\n{'-'*80}")
        print(f"🔍 SAMPLE {i}/{len(text_samples)} | MODEL: {model_name}")
        print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"⏰ Sample Start: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'-'*80}")
        
        try:
            # Create tasks for this text sample
            print(f"📋 Creating classification tasks...")
            tasks = create_classification_tasks(text, theme_generator_agent, theme_validator_agent)
            
            # Create crew with both agents
            print(f"🏗️ Creating crew with {len(tasks)} tasks...")
            crew = Crew(
                agents=[theme_generator_agent, theme_validator_agent],
                tasks=tasks,
                verbose=True,
                process="sequential"  # Tasks run in sequence for coordination
            )
            
            print(f"🚀 Executing collaborative classification...")
            start_time = time.time()
            
            # Execute the crew - NO RETRY LOGIC TO KEEP COSTS LOW
            result = crew.kickoff()
            
            execution_time = time.time() - start_time
            successful_count += 1
            
            # Extract theme for logging
            extracted_theme = extract_theme_from_result(str(result))
            
            print(f"\n✅ SUCCESS - Sample {i} completed!")
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            print(f"🎯 Extracted Theme: {extracted_theme}")
            print(f"📊 Progress: {successful_count} success, {failed_count} failed")
            
            # Store results
            sample_result = {
                "model": model_name,
                "sample_number": i,
                "input_text": text,
                "result": str(result),
                "extracted_theme": extracted_theme,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            results.append(sample_result)
            
        except Exception as e:
            failed_count += 1
            error_msg = str(e)
            
            print(f"\n❌ FAILED - Sample {i} error!")
            print(f"🚨 Error: {error_msg}")
            print(f"📊 Progress: {successful_count} success, {failed_count} failed")
            print(f"⚠️ NO RETRY - Continuing to next sample to minimize costs")
            
            error_result = {
                "model": model_name,
                "sample_number": i,
                "input_text": text,
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
            results.append(error_result)
        
        # Progress update every 10 samples
        if i % 10 == 0:
            elapsed = time.time() - model_start_time
            avg_time = elapsed / i
            estimated_remaining = avg_time * (len(text_samples) - i)
            print(f"\n📈 PROGRESS UPDATE - Sample {i}/{len(text_samples)}")
            print(f"⏱️ Elapsed: {elapsed:.1f}s | Avg: {avg_time:.1f}s/sample")
            print(f"🔮 Est. remaining: {estimated_remaining:.1f}s")
            print(f"📊 Success rate: {(successful_count/i)*100:.1f}%")
    
    model_total_time = time.time() - model_start_time
    
    print(f"\n{'='*60}")
    print(f"🏁 MODEL {model_name} COMPLETED")
    print(f"⏱️ Total time: {model_total_time:.2f} seconds")
    print(f"✅ Successful: {successful_count}/{len(text_samples)}")
    print(f"❌ Failed: {failed_count}/{len(text_samples)}")
    print(f"📈 Success rate: {(successful_count/len(text_samples))*100:.1f}%")
    print(f"⚡ Avg time per sample: {model_total_time/len(text_samples):.2f}s")
    print(f"{'='*60}")
    
    return {
        "model": model_name,
        "status": "completed",
        "total_time": model_total_time,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "results": results
    }

def extract_theme_from_result(result_text):
    """Extract the final theme from the result text"""
    try:
        # Look for FINAL_THEME in the result
        if "FINAL_THEME:" in result_text:
            lines = result_text.split('\n')
            for line in lines:
                if "FINAL_THEME:" in line:
                    return line.split("FINAL_THEME:")[1].strip()
        
        # Fallback: look for CLASSIFICATION
        if "CLASSIFICATION:" in result_text:
            lines = result_text.split('\n')
            for line in lines:
                if "CLASSIFICATION:" in line:
                    return line.split("CLASSIFICATION:")[1].strip()
        
        return "Unable to extract theme"
    except:
        return "Error extracting theme"

def load_test_data():
    """Load all records from input Excel file"""
    try:
        print("📂 Loading test data from excel...")
        df = pd.read_excel('<your input file>.xlsx', sheet_name='<sheet name>')
        
        # Extract text column
        test_samples = df['Text'].tolist()
        
        print(f"✅ Successfully loaded {len(test_samples)} records from Excel file")
        print(f"📊 Sample preview:")
        for i, sample in enumerate(test_samples[:3], 1):
            print(f"   {i}. {sample[:80]}...")
        if len(test_samples) > 3:
            print(f"   ... and {len(test_samples) - 3} more records")
        
        return test_samples
        
    except FileNotFoundError:
        print("❌ Error: xlsx file not found")
        print("🔄 Falling back to default test samples...")
        return [
            "<test sample tweet 1>",
            "<test sample tweet 2>"
        ]
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        print("🔄 Falling back to default test samples...")
        return [
            "<test sample tweet 1>",
            "<test sample tweet 2>"
        ]

def main():
    """Main execution function"""
    print("🤖 AgenticAI Multi-Model Collaborative Theme Classification Test - FULL DATASET")
    print("=" * 100)
    print("👥 Two-Agent Architecture for Each Model:")
    print("   🎯 Agent 1: Theme Classification Specialist")
    print("   ✅ Agent 2: Theme Validation Expert")
    print("=" * 100)
    
    # Load all test data from Excel file
    test_samples = load_test_data()
    
    print(f"🔬 Testing {len(MODELS_TO_TEST)} models with {len(test_samples)} records each")
    print(f"🧪 Total tests to execute: {len(MODELS_TO_TEST) * len(test_samples)}")
    print("⚠️  No retry logic - single attempt per test to minimize LLM costs")
    print("=" * 100)
    
    print(f"\n📝 Test dataset contains {len(test_samples)} records:")
    for i, sample in enumerate(test_samples[:5], 1):
        print(f"   {i}. {sample[:100]}...")
    if len(test_samples) > 5:
        print(f"   ... and {len(test_samples) - 5} more records")
    
    # Store all results
    all_results = []
    model_summaries = []
    
    # Test each model
    for model_name in MODELS_TO_TEST:
        try:
            model_result = run_collaborative_classification_for_model(model_name, test_samples)
            all_results.extend(model_result["results"])
            
            # Create model summary
            successful_results = [r for r in model_result["results"] if r.get("status") == "success"]
            failed_results = [r for r in model_result["results"] if r.get("status") == "error"]
            
            avg_time = 0
            if successful_results:
                avg_time = sum(r["execution_time"] for r in successful_results) / len(successful_results)
            
            model_summary = {
                "model": model_name,
                "total_samples": len(test_samples),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "average_time": avg_time,
                "total_time": model_result.get("total_time", 0),
                "status": model_result["status"]
            }
            model_summaries.append(model_summary)
            
            print(f"\n📊 {model_name} Summary:")
            print(f"   ✅ Successful: {len(successful_results)}/{len(test_samples)}")
            print(f"   ❌ Failed: {len(failed_results)}/{len(test_samples)}")
            print(f"   ⏱️ Avg time: {avg_time:.2f}s")
            
        except Exception as e:
            print(f"\n❌ Critical error testing {model_name}: {e}")
            model_summary = {
                "model": model_name,
                "total_samples": len(test_samples),
                "successful": 0,
                "failed": len(test_samples),
                "average_time": 0,
                "total_time": 0,
                "status": "critical_error",
                "error": str(e)
            }
            model_summaries.append(model_summary)
    
    # Generate comprehensive summary report
    print(f"\n{'='*100}")
    print("📊 COMPREHENSIVE MULTI-MODEL TEST SUMMARY")
    print(f"{'='*100}")
    
    total_tests = len(MODELS_TO_TEST) * len(test_samples)
    successful_tests = len([r for r in all_results if r.get("status") == "success"])
    failed_tests = len([r for r in all_results if r.get("status") == "error"])
    
    print(f"🔬 Total models tested: {len(MODELS_TO_TEST)}")
    print(f"📝 Samples per model: {len(test_samples)}")
    print(f"🧪 Total tests executed: {total_tests}")
    print(f"✅ Successful tests: {successful_tests}")
    print(f"❌ Failed tests: {failed_tests}")
    print(f"📈 Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    results_file = f'agenticai_multi_model_theme_classification_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            "test_info": {
                "timestamp": timestamp,
                "models_tested": MODELS_TO_TEST,
                "test_samples": test_samples,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": (successful_tests/total_tests)*100
            },
            "model_summaries": model_summaries,
            "detailed_results": all_results
        }, f, indent=2)
    
    # Create Excel report
    excel_file = f'agenticai_multi_model_theme_classification_results_{timestamp}.xlsx'
    
    # Prepare data for Excel
    excel_data = []
    for result in all_results:
        if result.get("status") == "success":
            extracted_theme = extract_theme_from_result(result["result"])
            excel_data.append({
                "Model": result["model"],
                "Sample_Number": result["sample_number"],
                "Input_Text": result["input_text"],
                "Extracted_Theme": extracted_theme,
                "Full_Result": result["result"],
                "Execution_Time": result["execution_time"],
                "Status": result["status"],
                "Timestamp": result["timestamp"]
            })
        else:
            excel_data.append({
                "Model": result["model"],
                "Sample_Number": result["sample_number"],
                "Input_Text": result["input_text"],
                "Extracted_Theme": "ERROR",
                "Full_Result": result.get("error", "Unknown error"),
                "Execution_Time": 0,
                "Status": result["status"],
                "Timestamp": result["timestamp"]
            })
    
    # Create summary data for Excel
    summary_data = []
    for summary in model_summaries:
        summary_data.append({
            "Model": summary["model"],
            "Total_Samples": summary["total_samples"],
            "Successful": summary["successful"],
            "Failed": summary["failed"],
            "Success_Rate": f"{(summary['successful']/summary['total_samples'])*100:.1f}%",
            "Average_Time": f"{summary['average_time']:.2f}s",
            "Total_Time": f"{summary['total_time']:.2f}s",
            "Status": summary["status"]
        })
    
    # Write to Excel
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        pd.DataFrame(excel_data).to_excel(writer, sheet_name='Detailed_Results', index=False)
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Model_Summary', index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   📄 JSON: {results_file}")
    print(f"   📊 Excel: {excel_file}")
    
    # Print model performance ranking
    print(f"\n🏆 MODEL PERFORMANCE RANKING:")
    successful_models = [s for s in model_summaries if s["successful"] > 0]
    successful_models.sort(key=lambda x: (x["successful"], -x["average_time"]), reverse=True)
    
    for i, model in enumerate(successful_models, 1):
        print(f"   {i}. {model['model']}: {model['successful']}/{model['total_samples']} success, {model['average_time']:.2f}s avg")
    
    failed_models = [s for s in model_summaries if s["successful"] == 0]
    if failed_models:
        print(f"\n❌ MODELS WITH NO SUCCESSFUL TESTS:")
        for model in failed_models:
            print(f"   • {model['model']}: {model.get('error', 'All tests failed')}")
    
    print(f"\n🎉 Multi-model testing completed!")
    print(f"  Check the Excel file for detailed analysis: {excel_file}")

if __name__ == "__main__":
    main()
