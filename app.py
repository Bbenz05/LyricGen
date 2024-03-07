"""
This module is designed to generate a dataset of responses using the OpenAI API and Streamlit for UI.
"""

import concurrent.futures
import json
import os
import random
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
    model_name = st.sidebar.text_input("Model:", value="ft:gpt-3.5-turbo-0125:personal:kye:8yrWwBuF")
    system_message = ''
    user_input = st.sidebar.text_area("Type your thoughts:", value="it's been a minute. You know, sometimes, I just find myself sitting here, staring at my phone, debating if I should hit her up or not. It's crazy, right? I mean, I genuinely don't enjoy being by myself too much. It gets to me, makes me think too much about us, about what we had. I keep thinking about dialing her number, wishing she'd just walk back through that door. But, bro, it's like I'm caught in this tug of war with myself. On one hand, I'm trying to keep my own space, trying to guard my heart 'cause, let's be real, I've been through the wringer with love. I'm not really looking for more pain, but then again, there's this twisted part of me that craves it, feels like it's the only way I feel something real, you know?\n\nAnd then, there's this thought that maybe, just maybe, I need a break from all this chaos in my heart. Maybe we both do. But it's complicated, man. We're so tangled up in each other, it's like we're too far gone to just have a casual thing and then act like nothing's happened. We're way past that point.\n\nIt's deep, bro. Like, really deep. We've shared too much, felt too much. There's no simple way to just walk away from what we've built, even if it's messed up at times. I find myself wishing I could just call her, tell her I'm coming home, but it's not that simple. We're stuck in this loop where letting go isn't an option, yet holding on feels just as impossible. It's like we're in this endless cycle, too in love to just quit, but too hurt to fully commit. It's exhausting, man.")
    number_of_responses = st.sidebar.text_input("Number of Responses:", value="10")
    if st.sidebar.button("Submit"):
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

    # Button to save the selected and edited responses
    if st.button("Create Dataset"):
        # Filter the responses to include only those that were selected
        selected_and_edited_responses = [resp for resp, selected in st.session_state['editable_responses'].values() if selected]
        
        # Use the selected and edited responses for the dataset
        dataset_jsonl = generate_dataset_jsonl(selected_and_edited_responses, system_message, user_input)
        st.code(dataset_jsonl, language='json')
        
        # Create a download button for the dataset
        st.download_button(
            label="Download Dataset as JSONL",
            data=dataset_jsonl,
            file_name='selected_responses.jsonl',
            mime='text/plain'
        )

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