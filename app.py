from flask import Flask,request,render_template
import pandas as pd

app= Flask(__name__) 
data= pd.read_csv('cleaned.csv')

@app.route('/')

def index():
    
    localityIds = sorted(data['localityId'].unique())
    return render_template('index.html',locations=localityIds)


if __name__=='__main__':
    app.run(debug=True)






