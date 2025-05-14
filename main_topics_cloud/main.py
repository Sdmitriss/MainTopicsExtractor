import warnings
import nltk
# import torch
import  numpy as np
import re
import pandas as pd
import os
import threading
import time
import sys

from pathlib import Path
from  module.dict_to_dataframe_parser import DictToDataFrameParser
from  module.bertopic_setup import InitBertopic
from  module.llm_cloud import TopicLLMll


from pathlib import Path
import json
from dotenv import load_dotenv,find_dotenv
# from typing import Optional

# from transformers import AutoTokenizer, pipeline
from pathlib import Path
from nltk.corpus import stopwords 
from IPython.display import display

warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings("ignore", category=UserWarning, module="bertopic")

# функция активности
stop_event = threading.Event()
def spin():
    i = True
    while not stop_event.is_set():
     if i:
        print("\rI compute ...",end='')
        i=False 
        time.sleep(2)
     else:
        print("\rwaiting ...",end='') 
        i=not i 
        time.sleep(2) 
       


# stop words для 'Vectorizers in BERTopic'
try:
    russian_stopwords = stopwords.words("russian") 
except:
    nltk.download("stopwords")
    russian_stopwords = stopwords.words("russian")
#  Перенесена в модуль llm_cloud
# load_dotenv(find_dotenv())

pattern  = r'[^\w,.\!? ]'
def cleaned_string(string: str, pattern: str = pattern) -> str: 
    '''
    По умолчанию  удаляются все символы, не являющиеся буквами, цифрами, пробелами и знаками 
                   препинания.
    Замена гиперссылок на  подстроку "гиперссылка"
    '''
    # Вопрос к ревью - замена  гипрссылок на ключевое слово гипрессылка стоит ли???
    string = re.sub(r'https?://\S+', 'гиперсссылка', string)
    return re.sub(pattern, '', string).strip()

#  Подготовка  текста без   учета  зависимости цепочек ответов (Гипотеза -тексты независмые)

def text_preparation_with_replay( df: pd.DataFrame, len_text: str = 3) -> tuple[list[int],list[str]]:
    ''' Подготавливает текстовые данные для передачи в BERTopic:
        применяет к каждому тексту функцию `cleaned_string`,
        и возвращает два списка:
        - список индексов для кластеризации,
        - очищенные строки (тексты), которые можно передать в модель.
        len_text -минимальная длина текста
    '''
    # Вопрос  ревью -каое значение len_text выбрать? (длина в символах не в словах!!!)
    if not isinstance(df, pd.DataFrame):
       raise TypeError("Expected DataFrame, got another type.")
    if  'text'not in df.columns:
       raise KeyError("Column 'text' is missing in the DataFrame")
   
    text = list(df[df['text'].str.len() >= len_text]['text'].items())
           
    
    return  map(list,zip(*text)) 



FOLDER_INPUT = 'input_data'
# Создадим папку  'input_data' в  родительском каталоге проекта
path_dir_input = (Path(__file__).resolve().parent.parent)/FOLDER_INPUT
path_dir_input.mkdir(parents=True, exist_ok=True)


## Парсим json
try:
    df_1 = DictToDataFrameParser(next(path_dir_input.glob('*.json')))
except:
    print(f'Проверь папку {FOLDER_INPUT} Возможно нет JSON-файла в папке.')
    sys.exit()
    


LEN_TEXT = 3 
index_df, text = text_preparation_with_replay(df_1.df, len_text= LEN_TEXT)
assert len(text) == len(index_df), (f'Lengths do not match: {len(text)} != {len(index_df)}')   

# модель для Embedding Models
# model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = "DeepPavlov/rubert-base-cased-conversational"
#параметры модели понижения размерности
umap_dict = dict(n_neighbors=3, n_components=5, min_dist=0.0, metric='cosine')
#параметры модели кластеризации
hdbscan_dict = dict(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
#параметры для Vectorizers
vectorizer_dict = dict(stop_words=russian_stopwords, ngram_range=(1, 3), min_df=1)
#параметры для TF_IDF
tf_idf_dict = dict(seed_words=["ссылка", "яндекс", "практикум", "стажировка", "курсы",'вакансия'],
                   seed_multiplier=2
                  )
                  
# nr_topics - количество  тем(ориентировочно)
# top_n_words - количество наиболее значимых слов для каждой темы.

NR_TOPIC = None
TOP_N_WORDS = 10
bertopic_model = InitBertopic(nr_topics=NR_TOPIC,
               top_n_words=TOP_N_WORDS,
               model=model,
               umap_dict=umap_dict, 
               hdbscan_dict=hdbscan_dict, 
               vectorizer_dict=vectorizer_dict, 
               tf_idf_dict=tf_idf_dict)


thread = threading.Thread(target=spin)
thread.start()
topics, probs = bertopic_model.topic_model.fit_transform(text)



# добавляем  в дтатфрейм( изменяем атрибут .df)
#столбцы topic - номер  кластера, probs - вероятность принадлежности к кластеру (важность?)
# Для зависмых текстов probs = 0 ( заглушка для дальнейшего отбора токенов d LLM, если  вокно не укладываемся)

if 'topic' in df_1.df.columns:
    df_1.df = df_1.df.drop(columns=['topic', 'probs'], errors='ignore')
df_1.df = df_1.df.join(pd.DataFrame({'topic': topics,'probs':np.round(probs,3)}, index=index_df))
df_1.df['probs'] = df_1.df['probs'].fillna(0)


# Для некотрых сообщений 'replay' дополнительная разметка по номеру кластера родительского сообщения.
while True:
    term = df_1.df['topic'].isna().sum()
    dict_1 = { i: df_1.df[df_1.df['topic'] == i]['message_id'].to_list() for i in  df_1.df.topic.dropna().unique()}
    for i in dict_1:
        df_1.df.loc[df_1.df['reply_to_message_id'].isin(dict_1[i]),'topic'] = i
    
    # Если количество NaN значений не изменилось, выходим из цикла
    if  term == df_1.df['topic'].isna().sum():
        break

stop_event.set()
thread.join()
print("Bertopic_OK")

system_message = '''Выдели основную тему из. текста.
Начни с краткого и ёмкого названия темы (от 5 до 10 слов), точно отражающего суть текста.
Затем выдели ключевые слова и фразы (5–10).
В конце сформулируй главную мысль.
Не добавляй лишнего текста.
Формат ответа 'Здесь укажиосновную тему'||'Здесь укажи ключевые слова'||'Здесь сформулируй главную мысль'
'''
user_message = user_message = '{text}'
llm_model =TopicLLMll(system_message, user_message)
topic_list = llm_model(df_1.df)

stop_event.clear()
thread = threading.Thread(target=spin)
thread.start()
topic_list = llm_model(df_1.df)

stop_event.set()
thread.join()
print("LLM_OK")



topic_list = [ (*i[0].split('||'), i[1])  for i in topic_list]

df_topic_list = pd.DataFrame(topic_list, columns=['Main Topic',' Keywords','Main Idea','topic']) \
    .set_index('topic') \
    .join(df_1.df['topic'].value_counts().rename('Topic_Frequency')) \
    .reset_index()
df_topic_list = df_topic_list.sort_values(by = 'Topic_Frequency', ascending=False)

df_final = df_1.df.merge(df_topic_list, how='left', on='topic')
df_final = df_final.dropna()

try:
    df_final[['topic','Topic_Frequency']] = df_final[['topic','Topic_Frequency']].astype('int')
    df_topic_list['topic'] = df_topic_list['topic'].astype('int')
except:
    pass



FOLDER_OUTPUT = 'output'
# Создадим папку  'output' в  родительском каталоге проекта
path_output = (Path(__file__).resolve().parent.parent)/FOLDER_OUTPUT
path_output.mkdir(parents=True, exist_ok=True)

def save_excel( df: pd.DataFrame, name_file: str, suffix: str) -> None:
   name = Path(name_file).with_suffix(suffix)
   path_file = path_output/name
   df.to_excel(path_file, index=False, engine='openpyxl')

def save_cluster_visual(model: InitBertopic,name_file: str) -> None:
    fig = bertopic_model.topic_model.visualize_topics(width = 1200, height = 800)
    name = Path(name_file).with_suffix('.html')
    path_file = path_output/name
    fig.write_html(path_file)

TOPIC='topic' 
CHAT_TOTAL ='chat_total'
SUFFIX = '.xlsx'    

save_excel(df_final, CHAT_TOTAL,SUFFIX)
save_excel(df_topic_list, TOPIC,SUFFIX)
save_cluster_visual(bertopic_model,TOPIC)
print('OK')