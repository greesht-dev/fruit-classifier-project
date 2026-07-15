Fruit classifier - AI analytics analysis tool

My why:
I teach kids how to build AI models as a Code Sensei at Code Ninjas. During a week-long AI Academy camp, I had students train fruit classifiers using Google's Teachable Machine and their webcams. Watching them work, I noticed clear patterns as someone new to the field of coding, tech, and AI myself. Some kids pushed beyond the given concept and experimented, others followed instructions exactly. Some understood why their model failed, others just kept testing randomly or as they go.
I built this to go deeper than what the kids were doing, and was encouraged by my boss. A fully instrumented version of the same project with a real model, real data pipeline, real analytics.

What it does:
Computer vision pipeline that classifies fruit images using a TensorFlow Lite model, logs every prediction with confidence scores and timestamps to CSV, and visualizes session data in a Streamlit dashboard.

Tech Stack
Python, TensorFlow Lite, Pandas, Matplotlib, Streamlit. Model trained with Google Teachable Machine.

How to Run
pip install -r requirements.txt
python3.11 app.py
streamlit run dashboard.py

Dashboard
Detection counts by fruit, confidence scores over time, summary stats per class.
