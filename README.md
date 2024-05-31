Steps to run Streamlit website 
Install Conda using a installer
Set Up Conda using the following command on terminal:
	conda init
Create a virtual environment:
	conda create --name myenv python=3.8
Activate the virtual Environment:
	conda activate myenv 
Clone the github repo in the environment:
	git clone https://github.com/D-Modi/Algobulls_Chatbot.git
Install pip:
	conda install pip
Go inside the directory and install requirments.txt
	cd Algobulls_Chatbot
	pip install -r requirements.txt
Run the streamlit file
	streamlit run scripts/csv_streamlit.py --server.enableXsrfProtection false

# In the code, we have taken the initial investment to be 150000  and the risk free rate to be 7% for all stratergies 

