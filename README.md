# QuickQuery - Advanced AI Student Helpdesk

## Overview
QuickQuery is an AI-powered student helpdesk application built with Streamlit and OpenAI GPT models. It provides real-time AI responses to student queries related to academics, campus facilities, events, and more. The app features smart query classification, analytics dashboard, multi-modal support, and API usage cost tracking.

## Features
- Real-time AI-powered query responses using OpenAI GPT-3.5/4
- Advanced query classification with confidence scoring
- Context-aware conversation handling with session history
- Interactive analytics dashboard with query statistics and visualizations
- Export conversation data to CSV for offline analysis
- Mobile-responsive design and user-friendly interface
- Quick action buttons for common queries
- API key configuration for enhanced AI responses

## Installation

1. Clone the repository or download the source code.

2. Create a Python virtual environment (recommended):

```bash
python -m venv venv
```

3. Activate the virtual environment:

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

4. Install the required dependencies:

```bash
pip install -r enhanced_requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run quickquery_final_app.py
```

2. Open your browser and navigate to:

```
http://localhost:8501
```

3. Enter your OpenAI API key in the sidebar to enable AI-powered responses.

4. Use the input box or quick query buttons to ask questions related to academics, campus facilities, events, and more.

5. View analytics and export conversation data as needed.

## Configuration

- API Key: Obtain your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys) and enter it in the sidebar.
- Model Selection: Choose between `gpt-3.5-turbo`, `gpt-4`, or `gpt-3.5-turbo-16k` models.
- Chat History Limit: Adjust the number of recent queries to retain in the session.
- Analytics: Toggle the analytics dashboard to view query statistics and trends.

## Troubleshooting

- Ensure you have an active internet connection for API calls.
- Verify your OpenAI API key is valid and has sufficient quota.
- If you encounter errors related to package versions, ensure dependencies are installed as per `enhanced_requirements.txt`.
- For accessibility warnings related to empty labels, the app has been updated to provide appropriate labels.

## License

This project is provided as-is for educational and demonstration purposes.

## Acknowledgments

- Built with Streamlit, OpenAI GPT, Plotly, and Pandas.
- OpenAI × NxtWave Buildathon 2025 Submission.
