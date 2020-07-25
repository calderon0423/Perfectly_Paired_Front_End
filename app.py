import sys
import os
import pandas as pd 
import tensorflow as tf 
import keras 
from keras.models import model_from_json, load_model
import numpy as np 
import pickle 
import flask 
from flask import Flask, render_template, jsonify, request, url_for, send_from_directory
from pymongo import MongoClient
from flask_restful import reqparse
import json



#read in csv file 
# df = pd.read_csv('/Users/angelicacalderon/repos/Perfectly_Paired/Resources/winemag-data-130k-v2.csv', low_memory=False)
# df = df.rename(columns = {'region_1': 'region_one', 'region_2': 'region_two'})
# df.to_csv('/Users/angelicacalderon/repos/Perfectly_Paired/Resources/winemag-data-130k-v2.csv',index=False)

# conn = 'mongodb://localhost:27017'
# client = MongoClient(conn)

# #connect to MongoDB wine_db
# db = client['wine_db']

# #create collections 
# reviews_collection = db['reviews']
# red_white_collection = db['red_white']

# #drop collection if available to remove duplicates
# db.reviews_collection.drop()
# db.red_white_collection.drop()

# #define function to read csv file as json 
# def csv_to_json(filename, header=True):
#     data = pd.read_csv(filename,low_memory=False)
#     return data.to_dict('records')

# #insert json file to collections 
# reviews_collection.insert_many(csv_to_json('Resources/winemag-data-130k-v2.csv'))
# red_white_collection.insert_many(csv_to_json('Resources/red_wine_combined.csv'))

# #connect to mongoDB wine db 
# mongoDB = ('mongodb:///?Server=localhost&Port=27017&Database=wine_db')


#################################################
# Flask Routes
#################################################


app = Flask(__name__)


# @app.route("/")
# def welcome():
#     """List all available api routes."""
#     return (
#         f"<h2>Welcome to the Directory</h2>"
#         f"Available Routes:<br/>"
#         f"<a href = '/reviewsdata' target='_blank'>/Wine Reviews Data</a><br/>"
#         f"<a href = '/red_white_wines_data' target='_blank'>/Red & White Wines Data</a><br/>"
#         f"<a href = '/predict_red_white' target='_blank'>/Red & White Prediction</a><br/>"
#         f"<a href = '/predict_type' target='_blank'>/Wine Type Prediction</a><br/>"
#     )

#routes to render .html pages 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/quality')
def quality():
    return render_template('quality.html')

@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

#@app.route('/reviews_2')
#def reviews_2():
#    return render_template('reviews_RG.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/red_wine')
def redWine():
    return render_template('red_wine.html')

@app.route('/white_wine')
def whiteWine():
    return render_template('white_wine.html')

@app.route('/reviewscode')
def reviewsCode():
    return render_template('reviewscode.html')

@app.route('/redwhitecode')
def redWhiteCode():
    return render_template('redwhitecode.html')

@app.route('/webdesign')
def webDesign():
    return render_template('webdesign.html')

@app.route('/reviewscodeRG')
def reviewsCodeRG():
    return render_template('reviewscodesRG.html')

@app.route('/flaskcode')
def flaskCode():
    return render_template('flask.html')
 
#@app.route('/reviewsdata')
#def reviewsData():
#    reviews_documents = reviews_collection.find()
#    response = []
#    for document in reviews_documents:
#        document['_id'] = str(document['_id'])
#        response.append(document)
#    # 
#    return jsonify(response)
#
##red_white_wines route 
#@app.route('/red_white_wines_data')
#def redWhiteData():
#    red_white_documents = red_white_collection.find()
#    response = []
#    for document in red_white_documents:
#        document['_id'] = str(document['_id'])
#        response.append(document)
#    return jsonify(response)

#wine list for drop down menu to use with WordCloud 
@app.route('/winelist')
def changename():
    with open('./Resources/variety_adj.json') as f:
        data = json.load(f)
    return json.dumps(data)

#load json file to create WordCloud 
@app.route('/wordcloud')
def name():
    with open('./Resources/frequency_dict.json') as f:
        data = json.load(f)
    return json.dumps(data)

#load json file to create bar chart 
@app.route('/barchart')

def barchart():

    parser = reqparse.RequestParser()
    parser.add_argument('variety', type=str, required=True, help="Provide a phrase to rate", action='append')
    args = parser.parse_args()
    variety = args['variety']

    with open('./Resources/winemag-data-barchart.json') as f:
        data = json.load(f)
        data = [x for x in data if x['variety']==variety[0]]
    return json.dumps(data)

#load json file to create bubble chart 
@app.route('/bubblechart')
def bubblechart():

    parser = reqparse.RequestParser()
    parser.add_argument('variety', type=str, required=True, help="Provide a phrase to rate", action='append')
    args = parser.parse_args()
    variety = args['variety']

    with open('./Resources/winemag-data-bubblechart.json') as f:
        data = json.load(f)
        data = [x for x in data if x['variety']==variety[0]]
    return json.dumps(data)

#load province json file 
@app.route('/province')
def province():
    with open('./Resources/Country_Province.json') as f:
        data = json.load(f)
    return json.dumps(data)

#predict wine type route - Naive Model 
@app.route('/predict_type')
def predictType():
    #user input 
    parser = reqparse.RequestParser()
    parser.add_argument('adjectives', type=str, required=True, help="This is expecting a selection of wine adjectives", action='append')
    args = parser.parse_args()
    adjectives = args['adjectives']
    print(adjectives)

    #load model .h5 files 
    vectorizer_file = "tokenizer.h5"
    tokenizer_file = "vectorizer.h5"
    NBModel = 'sentiment_scoring.h5'
    # NBModel = 'dl_v2.h5'


    vectorizer = pickle.load(open('Naive_sentiment_model/'+vectorizer_file, 'rb'))
    tokenizer = pickle.load(open('Naive_sentiment_model/'+tokenizer_file, 'rb'))
    nbModel = pickle.load(open('Naive_sentiment_model/'+NBModel, 'rb'))
    # nbModel = load_model('Naive_sentiment_model/dl_v2.h5')

    
    # run model with user_input argument 
    user_input=adjectives
    X_new = vectorizer.transform(user_input)
    X_new = tokenizer.transform(X_new)
    result = nbModel.predict(X_new)
    return jsonify({'wine_type': result[0]})

    # user_input=adjectives
    # X_new = vectorizer.transform(user_input)
    # X_new = tokenizer.transform(X_new)
    # result = nbModel.predict_classes(X_new)
    # result = pd.DataFrame(result,columns=['Class'])
    # return jsonify({'wine_type': result[0]})

#predict Red/White wine based on properties - Deep Learning Model 
@app.route('/predict_quality')
def redwhitepredict():
    # user input
    parser = reqparse.RequestParser()
    parser.add_argument('characteristics', type=str, required=True, help="This is expecting a selection of wine characteristics", action='append')
    args = parser.parse_args()
    characteristics = args['characteristics'][0].split(' ')

    #  Load the model
    redorwhite_model = load_model('Red_and_White_Analysis/redorwhite_model_trained.h5')

    # #predict the wine class based on model and save output to 'out'
    user_input_characteristics=[float(i) for i in characteristics]
    print(user_input_characteristics)

    #run model with user input
    result_characteristics = redorwhite_model.predict_classes([user_input_characteristics])
    result_characteristics = "White" if result_characteristics[0] ==1 else "Red"
    return jsonify({'wine_selection': result_characteristics})


#@app.route('/predict_wine_variety')
#def predict_variety():
#    #user input 
#    parser = reqparse.RequestParser()
#    parser.add_argument('adjectives', type=str, required=True, help="This is expecting a selection of wine adjectives", action='append')
#    args = parser.parse_args()
#    adjectives = args['adjectives']
#    parser.add_argument('country', type=str, required=True, help="This is expecting one selection of country", action='append')
#    args = parser.parse_args()
#    country = args['country']
#    parser.add_argument('province', type=str, required=True, help="This is expecting one selection of province", action='append')
#    args = parser.parse_args()
#    country = args['province']
#    parser.add_argument('points_grouped', type=str, required=True, help="This is expecting one selection of province", action='append')
#    args = parser.parse_args()
#    points_grouped = args['points_grouped']
#
#    user_input=[]
#    for adj in adjectives:
#        user_input.append(adj)
#    for c in country:
#        user_input.append(c)
#    for p in province:
#        user_input.append(p)
#
#    variables = ['White', 'Red', 'ripe', 'crisp', 'mature', 'tropical', 'rich', 'sweet', 'fresh', 'honeyed', 'fruity', 'smooth',
#    'soft', 'bright', 'dry', 'earthy', 'rubbery', 'savory', 'vanilla', 'bitter', 'intense', 'traditional', 'nutty', 'Argentina',
#    'Australia', 'Austria', 'Brazil', 'Bulgaria', 'Canada', 'Chile', 'Croatia', 'Cyprus', 'Czech Republic', 'Egypt', 'England', 
#    'France', 'Georgia', 'Germany', 'Greece', 'Hungary', 'India', 'Israel', 'Italy', 'Lebanon', 'Luxembourg', 'Macedonia', 'Mexico',
#    'Moldova', 'Morocco', 'New Zealand', 'Peru', 'Portugal', 'Romania', 'Serbia', 'Slovakia', 'Slovenia', 'South Africa', 'Spain',
#    'Switzerland', 'Turkey', 'US', 'Ukraine', 'Uruguay', 'Achaia', 'Aconcagua Costa', 'Aconcagua Valley', 'Aegean', 'Agioritikos',
#    'Ahr', 'Alentejano', 'Alentejo', 'Alsace', 'America', 'Andalucia', 'Apalta', 'Arcadia', 'Arizona', 'Atalanti Valley', 'Atlantida',
#    'Attica', 'Australia Other', 'Austria.1', 'Awatere Valley', 'Baden', 'Bairrada', 'Beaujolais', 'Beira Interior', 'Beiras', 
#    'Bekaa Valley', 'Black Sea Coastal', 'Bordeaux', 'Bot River', 'Brazil.1', 'Brda', 'Breedekloof', 'British Columbia', 'Buin', 
#    'Bulgaria.1', 'Burgenland', 'Burgundy', 'Bío Bío Valley', 'Cachapoal Valley', 'Cahul', 'California', 'Campanha', 'Canelones', 
#    'Canterbury', 'Cape Agulhas', 'Cape Peninsula', 'Cape South Coast', 'Cappadocia', 'Carnuntum', 'Casablanca & Leyda Valleys',
#    'Casablanca Valley', 'Casablanca-Curicó Valley', 'Catalonia', 'Cauquenes Valley', 'Cederberg', 'Central Italy', 'Central Otago',
#    'Central Spain', 'Central Valley', 'Cephalonia', 'Ceres Plateau', 'Chalkidiki', 'Champagne', 'Chile.1', 'Coastal Region', 'Coelemu',
#    'Colares', 'Colchagua Costa', 'Colchagua Valley', 'Colorado', 'Commandaria', 'Connecticut', 'Constantia', 'Corinth', 'Crete',
#    'Croatia.1', 'Curicó Valley', 'Curicó and Leyda Valleys', 'Curicó and Maipo Valleys', 'Cyclades', 'Cyprus.1', 'Dalmatian Coast',
#    'Dan', 'Danube River Plains', 'Darling', 'Dealu Mare', 'Dealurile Hușilor', 'Dealurile Munteniei', 'Devon Valley', 'Dingač', 
#    'Dolenjska', 'Douro', 'Drama', 'Durbanville', 'Duriense', 'Dão', 'East Coast', 'Eger', 'Egypt.1', 'Eilandia', 'Elazığ', 
#    'Elazığ-Diyarbakir', 'Elgin', 'Elim', 'Ella Valley', 'Elqui Valley', 'England.1', 'Epanomi', 'Estremadura', 'Florina', 'France Other', 
#    'Franken', 'Franschhoek', 'Galicia', 'Galil', 'Galilee', 'Georgia.1', 'Germany.1', 'Gisborne', 'Gladstone', 'Golan Heights',
#    'Goriska Brda', 'Goumenissa', 'Greece.1', 'Groenekloof', 'Halkidiki', "Hawke's Bay", 'Hemel en Aarde', 'Hvar', 'Ica', 'Idaho', 
#    'Iowa', 'Ismarikos', 'Israel.1', 'Istria', 'Italy Other', 'Itata Valley', 'Jerusalem Hills', 'Jidvei', 'Jonkershoek Valley', 
#    'Juanico', 'Judean Hills', 'Kakheti', 'Kamptal', 'Kathikas', 'Kentucky', 'Korinthia', 'Krania Olympus', 'Kremstal', 'Kumeu',
#    'Kutjevo', 'Lakonia', 'Landwein Rhein', 'Languedoc-Roussillon', 'Lebanon.1', 'Leithaberg', 'Lesbos', 'Letrinon', 'Levante', 
#    'Leyda Valley', 'Leyda Valley-Maipo Valley', 'Limarí Valley', 'Lisboa', 'Loire Valley', 'Lolol Valley', 'Lombardy', 'Loncomilla Valley', 
#    'Lontué Valley', 'Lutzville Valley', 'Macedonia.1', 'Maipo Valley', 'Maipo Valley-Colchagua Valley', 'Malleco', 'Mantinia', 
#    'Marchigue', 'Markopoulo', 'Marlborough', 'Martinborough', 'Massachusetts', 'Maule Valley', 'Mendoza Province', 'Michigan',
#    'Middle and South Dalmatia', 'Minho', 'Missouri', 'Mittelburgenland', 'Mittelrhein', 'Moldova.1', 'Molina', 'Monemvasia', 'Montevideo',
#    'Moravia', 'Morocco.1', 'Mosel', 'Mosel-Saar-Ruwer', 'Moselle Luxembourgeoise', 'Mount Athos', 'Murfatlar', 'Muzla', 'Nahe', 
#    'Naoussa', 'Nashik', 'Negev', 'Negev Hills', 'Nelson', 'Nemea', 'Neuchâtel', 'Nevada', 'New Jersey', 'New Mexico', 'New South Wales',
#    'New York', 'New Zealand.1', 'Niederösterreich', 'North Carolina', 'North Dalmatia', 'Northeastern Italy', 'Northern Cape',
#    'Northern Spain', 'Northwestern Italy', 'Ohio', 'Olifants River', 'Ontario', 'Oregon', 'Other', 'Overberg', 'Paardeberg',
#    'Paarl', 'Pafos', 'Pageon', 'Palmela', 'Panciu', 'Pangeon', 'Patras', 'Peljesac', 'Peloponnese', 'Pennsylvania', 'Península de Setúbal',
#    'Peumo', 'Pfalz', 'Philadelphia', 'Piedmont', 'Piekenierskloof', 'Pinto Bandeira', 'Pirque', 'Pitsilia Mountains', 'Podunavlje',
#    'Portuguese Table Wine', 'Primorska', 'Progreso', 'Provence', 'Puente Alto', 'Rapel Valley', 'Rapsani', 'Recas', 'Requinoa', 
#    'Retsina', 'Rheingau', 'Rheinhessen', 'Rhône Valley', 'Ribatejano', 'Rio Claro', 'Robertson', 'Romania.1', 'Sagrada Familia',
#    'Samson', 'San Antonio', 'San Antonio de las Minas Valley', 'San Clemente', 'San Jose', 'San Vicente', 'Santa Cruz', 'Santorini', 
#    'Sebes', 'Serra Gaúcha', 'Serra do Sudeste', 'Shomron', 'Sicily & Sardinia', 'Simonsberg-Paarl', 'Simonsberg-Stellenbosch',
#    'Slovenia.1', 'Slovenska Istra', 'Sopron', 'South Africa.1', 'South Australia', 'South Island', 'Southern Italy', 'Southwest France',
#    'Spain Other', 'Spanish Islands', 'Steiermark', 'Stellenbosch', 'Sterea Ellada', 'Swartland', 'Switzerland.1', 'Szekszárd',
#    'Südburgenland', 'Südoststeiermark', 'Südsteiermark', 'Tarnave', 'Tasmania', 'Tejo', 'Texas', 'Thermenregion', 'Thessalikos',
#    'Thrace', 'Thracian Valley', 'Ticino', 'Tikves', 'Tokaj', 'Tokaji', 'Traisental', 'Tulbagh', 'Turkey.1', 'Tuscany', 'Távora-Varosa', 
#    'Ukraine.1', 'Upper Galilee', 'Urla-Thrace', 'Uruguay.1', 'Vale Trentino', 'Vale dos Vinhedos', 'Valle de Guadalupe', 'Veneto', 
#    'Vermont', 'Victoria', 'Vienna', 'Viile Timisului', 'Villány', 'Vin de Pays de Velvendo', 'Vinho Espumante de Qualidade',
#    'Vinho Verde', 'Vipavska Dolina', 'Virginia', 'Vânju Mare', 'Wachau', 'Wagram', 'Wagram-Donauland', 'Waiheke Island', 'Waipara',
#    'Waipara Valley', 'Wairarapa', 'Wairau Valley', 'Waitaki Valley', 'Walker Bay', 'Washington', 'Washington-Oregon', 'Weinland Österreich',
#    'Weinviertel', 'Wellington', 'Western Australia', 'Western Cape', 'Wiener Gemischter Satz', 'Württemberg', 'Zenata', 'Österreichischer Perlwein',
#    'Österreichischer Sekt', 'Štajerska', 'Župa']
#
#    input_dict = {}
#
#    for name in variables:
#        if name in user_input:
#            input_dict[name] = [1]
#        else:
#            input_dict[name] = [0]
#        
#    input_dict['points_grouped'] = [points_grouped]
#    
#    input_df = pd.DataFrame.from_dict(input_dict, orient='columns')
#
#
#    #  Load the model
#    variety_model = load(open('Variety_model/variety_rf.sklearn', 'rb'))
#
#    # Load scalar
#    X_scaler = load(open('Variety_model/X_scaler.sklearn', 'rb'))
#
#    #predict the wine class based on model and save output to 'out'
#    input_scaled = X_scaler.transform(input_df)
#    out = variety_model.predict(input_scaled)
#    out = out[0]
#    return jsonify({'wine_selection': str(out)})


if __name__ == "__main__":
    app.run(debug=True)





