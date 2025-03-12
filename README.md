# Project Setup Guide

## Prerequisites
Ensure you have Python installed on your system. If not, download and install it from [python.org](https://www.python.org/).

## Steps to Run the Code

### 1. Get a Hugging Face Token
- Sign up or log in to [Hugging Face](https://huggingface.co/) to obtain your API token.
- Store the API token in a `.env` file with the following format:
  ```
  HF_TOKEN=your_huggingface_token_here
  ```

### 2. Install Dependencies
- Install the required packages using:
  ```
  pip install -r requirements.txt
  ```

### 3. Run the Application
- Execute the following command to start the application:
  ```
  streamlit run medibot.py
  ```

This will launch the application using Streamlit, and you can interact with it through your web browser.

## Additional Notes
- Ensure your `.env` file is in the same directory as `medibot.py`.
- If you encounter any issues, check if all dependencies are installed correctly.

Enjoy using MediBot! 🚀

