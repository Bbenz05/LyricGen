"""
This module is designed to generate a dataset of responses using the OpenAI API and Streamlit for UI.
"""

import concurrent.futures
import json
import os
import random
import requests
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # This loads the environment variables from .env file if it exists
apikey= st.sidebar.text_input("Api Key:")
client = OpenAI(api_key=apikey)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.sidebar.title('LyricGen')

    # Initialize session state for storing selected responses and responses
    if 'selected_responses' not in st.session_state:
        st.session_state['selected_responses'] = []
    if 'responses' not in st.session_state:
        st.session_state['responses'] = []
    if 'editable_responses' not in st.session_state:  # Initialize editable responses
        st.session_state['editable_responses'] = {}

    # Create a text area widget in the sidebar
    model_name = "ft:gpt-3.5-turbo-0125:personal:kye:8yrWwBuF"
    system_message = "You are a Large Language Model trained specfically to generate original lyrics based on any user's query. You were trained to only generate lyrics guaranteed to be catchy, memorable, meaningful and create a hit. You are only trained to generate lyrics in the style of Rylo Rodriguez, Summer Walker, NoCap, Bryson Tiller, Seddy Hendrinx, Rod Wave, Hunxho, CEO Trayle, Dee Baby, and Lil Baby."
    user_input = st.sidebar.text_area("Type your thoughts:")
    number_of_responses = st.sidebar.text_input("Number of Responses:", value="10")
    if st.sidebar.button("Submit", key="sidebar_submit"):
        st.session_state['responses'] = []  # Reset responses on new submission
        st.session_state['editable_responses'] = {}  # Reset editable responses
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in range(int(number_of_responses)):
                future = executor.submit(request, system_message, user_input, model_name)
                futures.append(future)
                
            for future in concurrent.futures.as_completed(futures):
                try:
                    message_content, temperature = future.result()  # Unpack the returned tuple
                    st.session_state['responses'].append((message_content, temperature))  # Store the message content and temperature as a tuple
                    st.session_state['editable_responses'][message_content] = message_content  # Initialize editable response with original content
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")
                    break  # Exit the loop on failure

    # Use text areas to allow editing of responses and checkboxes to select responses
    for i, (response, temperature) in enumerate(st.session_state['responses']):
        # Provide a checkbox for selecting the response
        is_selected = st.checkbox(f"Response {i+1}", key=f"select_response_{i}")
        # Provide a text area for editing the response
        edited_response = st.text_area(f"Temp: {temperature}", value=response, key=f"editable_response_{i}")
        
        # Update the session state with the edited response and selection status
        st.session_state['editable_responses'][response] = (edited_response, is_selected)

    # Conditionally render the "Submit" button based on whether there are any responses
    if st.session_state['responses']:  # This checks if the list of responses is not empty
        if st.button("Submit", key="main_submit"):
            # Filter the responses to include only those that were selected
            selected_and_edited_responses = [resp for resp, selected in st.session_state['editable_responses'].values() if selected]
            
            # Use the selected and edited responses for the dataset
            dataset_jsonl = generate_dataset_jsonl(selected_and_edited_responses, system_message, user_input)
            
            # Define your Discord webhook URL
            webhook_url = 'https://discord.com/api/webhooks/1215162518180200500/e6R2vp1ujtcYmMXryj8f-0N81hKz6leZejnZGXjMkE3HonXq3jayG5TUBAk145bZv8I2'
            
            # Prepare the content to be sent. You might want to customize this part.
            data = {
                "content": "Here are the selected and edited lyrics:",
                "username": "LyricGen Bot"
            }
            
            # Attach the dataset as a file. Discord expects files in a list of tuples [(filename, content)]
            files = {
                'file': ('selected_responses.jsonl', dataset_jsonl, 'application/json')
            }
            
            # Make a POST request to the Discord webhook URL with the data and file
            response = requests.post(webhook_url, data=data, files=files)
            
            # Check if the request was successful
            if response.status_code == 204:
                st.success("Sent!")
            else:
                st.error(f"Failed to send. Status code: {response.status_code}")

def request(system_message, user_input, model_name):
    """
    Sends a request to the OpenAI API and returns the response and temperature.
    
    :param system_message: The system message to send.
    :param user_input: The user input to send.
    :param model_name: The model name to use for the request.
    :return: A tuple containing the message content and temperature.
    """
    temperature = round(random.uniform(0, 1), 1)  # Set temperature to a random value between 0 and 1 with 1 decimal
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "system",
            "content": system_message
        }, {
            "role": "user",
            "content": user_input
        }],
        temperature=temperature,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message_content = response.choices[0].message.content
    return message_content, temperature # Return the message content

def generate_dataset_jsonl(selected_responses, system_message, user_input):
    """
    Generates a JSON Lines string from the selected responses.
    
    :param selected_responses: The responses selected for inclusion in the dataset.
    :param system_message: The system message associated with the responses.
    :param user_input: The user input associated with the responses.
    :return: A string in JSON Lines format containing the dataset.
    """
    datasets = []
    for response in selected_responses:
        dataset = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ]
        }
        datasets.append(json.dumps(dataset))
    return '\n'.join(datasets)

if __name__ == "__main__":
    main()
    
