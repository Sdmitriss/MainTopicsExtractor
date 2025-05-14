Проект MainTopicsExtractor  реализует  поиск главных тем сообщений чатов.
Реализация включает три этапа:
1. Преобразование файла формата *.json (выгрузка сообщений чата с сопутствующими метаданными — имя пользователя, дата и т.д.) в табличный вид.
За это отвечает модуль dict_to_dataFrame_parser.py (подробное описание содержится в докстринге модуля).

2. Кластеризация текстовых сообщений с помощью BERTopic — модуль bertopic_setup.py.
Используется модель эмбеддингов: "DeepPavlov/rubert-base-cased-conversational".
Подробное описание — в докстрингах модуля.

3.    Выделение главных тем с помощью LLM-модели. Реализовано два варианта:
  - Облачный вариант — находится в каталоге main_topics_cloud, используется модель ChatYandexGPT. Реализация представлена в модуле llm_cloud.py.

  - Локальный вариант — находится в каталоге main_topics, используется модель YandexGPT-5-Lite-8B-instruct-Q8_0-GGUF.
<pre><code>
# Структура  проекта для локальной LLM
project_root/ 
├── main_topics/
│   ├── src/
│   │   └── module/
│   │       ├── __init__.py
│   │       ├── dict_to_dataframe_parser.py
│   │       └── bertopic_setup.py
│   ├── research/
│   │   └── bertopic_add_LLM_localy.ipynb
│   ├── main.py
├── input_data/ (.json)
├── output/   ( .xlsx)
</code></pre>
 
<pre><code>
 # Структура  проекта для облачной LLM
project_root/ 
├── main_topics/
│   |── .env
│   │── module/
│   │ ├── __init__.py
│   │ ├── dict_to_dataframe_parser.py
│   │ |── bertopic_setup.py
|   | └── llm_cloud.py    -    
|   |
│   ├── research/
│   │   |── bertopic_add_LLM.ipynb
|   |   |── bertopic_baseline.ipynb
|   |   └── json_to_df.ipynb
│   ├── main.py
├── input_data/ (.json)
├── output/   ( .xlsx)
</code></pre>

Файлы json_to_df.ipynb и bertopic_baseline.ipynb содержат исследовательскую разработку модулей dict_to_dataframe_parser.py и bertopic_setup.py.

Файлы bertopic_add_LLM.ipynb, notebook.ipynb и bertopic_add_LLM_localy.ipynb — это исследовательская разработка кода для выделения главных тем с помощью LLM-моделей (облачной и локальной).

Инструкция по запуску сервиса(обе реализации):

1. Помести файл *.json в папку input_data (если папки нет — запусти main.py, она создастся).
2. Запусти: python main.py
3. Результаты будут в папке output:
   - topic.xlsx — выделенные темы,
   - chat_total.xlsx — полная обработка.

Корректная работа  main_topics_cloud требует наличия файла .env с переменными окружения:
- YC_API_KEY='........'       ключ доступа к YandexGPT
- FOLDER_ID='.........'        ID папки в Yandex Cloud










