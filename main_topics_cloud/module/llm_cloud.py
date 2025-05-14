import os
import tiktoken
import pandas as pd
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

class TopicLLMll():
    def __init__(self, task: str, format_user_messages: str, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.task = task
        self.format = format_user_messages
        if not os.getenv('FOLDER_ID') and os.getenv('YC_API_KEY'):
            raise  ValueError("Both environment variables must be set: FOLDER_ID and YC_API_KEY")
            
    system_message ='{task}'
    user_message = '{text}\n {format_user_messages}'
    user_message = user_message = "{text} "
    
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_message),('user',user_message)])
   
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    
    chat_model = ChatYandexGPT(
            folder_id=os.getenv("FOLDER_ID"),
            model_uri=f"gpt://{os.getenv('FOLDER_ID')}/yandexgpt-32k/latest"
         )
        
   
        

    def __call__(self, df_top: pd.DataFrame):
        df_top =df_top.dropna()
        df_top = df_top.sort_values(by='probs', ascending=False)
        topic=[]
        for i in df_top['topic'].unique():
            
            text = ' '.join(df_top[df_top['topic']== i ]['text'].to_list())
            text = self.count_cut_text(text, self.max_tokens)
            
            prompt = self.prompt_template.invoke({'task': self.task, 'text':text})
            topic.append((self.chat_model.invoke(prompt).content, i))
        return topic
                         
    
    def count_cut_text(self, text, max_tokens ):
        tokens = self.enc.encode(text)[:max_tokens]
        return self.enc.decode(tokens)
    