import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import time

def main():
    print("Starting construction activity matching...")
    
    # Load API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    print("✓ OpenAI client initialized")
    
    # Read Excel files
    input_df = pd.read_excel('SUBWAY TEST.xlsx')
    reference_df = pd.read_excel('detailed BoQ .xls')
    print(f"✓ Loaded {len(input_df)} input descriptions and {len(reference_df)} reference descriptions")
    
    # Prepare data
    input_descriptions = []
    input_indices = []
    
    for idx, row in input_df.iterrows():
        desc = str(row['Description']).strip()
        if desc and desc != 'nan':
            input_descriptions.append(desc)
            input_indices.append(idx)
    
    reference_descriptions = reference_df['Description'].dropna().tolist()
    
    # Create comprehensive prompt for single API call
    input_list = "\n".join([f"INPUT_{i+1}: {desc}" for i, desc in enumerate(input_descriptions)])
    ref_list = "\n".join([f"REF_{i+1}: {desc}" for i, desc in enumerate(reference_descriptions)])
    
    prompt = f"""You are a construction BOQ activity matching expert. Compare each INPUT description with ALL REFERENCE descriptions to find semantic matches, one thing in mind that numerical value like ratio , thickness etc. should be same(otherwise it will not remain as same activity) also  (same construction activity with different wording, synonyms, typos allowed)analyze both side i don't want just simmiller i want exact same if it is (the activity must be same ).

INPUT DESCRIPTIONS TO MATCH:
{input_list}

REFERENCE DESCRIPTIONS:
{ref_list}

TASK: For each INPUT, find the best matching REFERENCE (if any). Only match if they describe the SAME construction activity.

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "INPUT_1": {{"matched": true, "reference_id": "REF_5", "reference_text": "exact reference description here"}},
  "INPUT_2": {{"matched": false, "reference_id": "", "reference_text": ""}},
  "INPUT_3": {{"matched": true, "reference_id": "REF_2", "reference_text": "exact reference description here"}}
}}

Be precise. Only semantic matches for identical construction activities."""

    print("🔄 Sending batch request to OpenAI...")
    
    try:
        # Try models in order of cost (cheapest first)
        models = ["deepseek-chat"]
        
        response = None
        for model in models:
            try:
                print(f"Trying model: {model}")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a cafe interior construction expert. Respond only in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=8000,
                    temperature=0.1
                )
                print(f"✓ Success with {model}")
                break
            except Exception as e:
                print(f"Failed with {model}: {str(e)[:100]}...")
                continue
        
        if not response:
            raise Exception("All models failed - check your OpenAI quota and billing")
        
        result_text = response.choices[0].message.content.strip()
        print("✓ Received response from OpenAI")
        
        # Clean and parse JSON
        if result_text.startswith('```json'):
            result_text = result_text.split('```json')[1].split('```')[0].strip()
        elif result_text.startswith('```'):
            result_text = result_text.split('```')[1].split('```')[0].strip()
        
        matches = json.loads(result_text)
        print("✓ Parsed matching results")
        
        # Initialize result columns
        input_df['Matched'] = 'No'
        input_df['Matched Description'] = ''
        
        # Apply matches
        matched_count = 0
        for i, original_idx in enumerate(input_indices):
            input_key = f"INPUT_{i+1}"
            if input_key in matches:
                match_info = matches[input_key]
                if match_info.get("matched", False):
                    input_df.at[original_idx, 'Matched'] = 'Yes'
                    input_df.at[original_idx, 'Matched Description'] = match_info.get("reference_text", "")
                    matched_count += 1
        
        # Save results
        output_file = 'output_matched_gpt4.xlsx'
        input_df.to_excel(output_file, index=False)
        
        print(f"✅ COMPLETE!")
        print(f"📊 Results: {matched_count}/{len(input_descriptions)} descriptions matched")
        print(f"💾 Output saved to: {output_file}")
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON response: {e}")
        print("Creating output file with no matches...")
        input_df['Matched'] = 'No'
        input_df['Matched Description'] = ''
        input_df.to_excel('output_matched_gpt4.xlsx', index=False)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        input_df['Matched'] = 'No'
        input_df['Matched Description'] = ''
        input_df.to_excel('output_matched_gpt4.xlsx', index=False)

if __name__ == "__main__":
    main()