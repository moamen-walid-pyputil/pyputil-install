"""
Utility functions for the auto-installer.

Provides stdlib detection, comprehensive import-to-pip name mapping,
system module detection, and skip logic used by both sync and async
installers. Contains exhaustive mappings collected from PyPI, Python
documentation, and real-world package analysis.
"""

import sys
import os
import importlib.util
from typing import FrozenSet, Dict, Optional, Set


# ===========================================================================
# Comprehensive import name → pip package name mapping
# ===========================================================================
# Sources:
#   - PyPI top 8000 packages analysis
#   - Python Packaging User Guide
#   - Common StackOverflow import errors
#   - Framework documentation (Django, Flask, TensorFlow, etc.)
# ===========================================================================

_PIP_NAME_MAP: Dict[str, str] = {
    # ── Web & HTTP ──────────────────────────────────────────────────────
    'bs4': 'beautifulsoup4',
    'beautifulsoup': 'beautifulsoup4',
    'BeautifulSoup': 'beautifulsoup4',
    'httpx': 'httpx',
    'aiohttp': 'aiohttp',
    'requests': 'requests',
    'urllib3': 'urllib3',
    'certifi': 'certifi',
    'chardet': 'chardet',
    'charset_normalizer': 'charset-normalizer',
    'idna': 'idna',
    'socks': 'PySocks',
    'sockshandler': 'PySocks',
    'websocket': 'websocket-client',
    'websockets': 'websockets',
    'socketio': 'python-socketio',
    'engineio': 'python-engineio',
    'flask': 'Flask',
    'flask_cors': 'Flask-Cors',
    'flask_sqlalchemy': 'Flask-SQLAlchemy',
    'flask_migrate': 'Flask-Migrate',
    'flask_login': 'Flask-Login',
    'flask_wtf': 'Flask-WTF',
    'flask_mail': 'Flask-Mail',
    'flask_restful': 'Flask-RESTful',
    'flask_restx': 'flask-restx',
    'flask_socketio': 'Flask-SocketIO',
    'flask_admin': 'Flask-Admin',
    'flask_babel': 'Flask-Babel',
    'flask_cache': 'Flask-Caching',
    'flask_compress': 'Flask-Compress',
    'flask_limiter': 'Flask-Limiter',
    'flask_jwt_extended': 'Flask-JWT-Extended',
    'flask_marshmallow': 'flask-marshmallow',
    'django': 'Django',
    'djangorestframework': 'djangorestframework',
    'rest_framework': 'djangorestframework',
    'django_filters': 'django-filter',
    'django_cors_headers': 'django-cors-headers',
    'django_extensions': 'django-extensions',
    'django_debug_toolbar': 'django-debug-toolbar',
    'django_celery': 'django-celery',
    'django_allauth': 'django-allauth',
    'django_ckeditor': 'django-ckeditor',
    'django_cleanup': 'django-cleanup',
    'django_compressor': 'django-compressor',
    'django_crispy_forms': 'django-crispy-forms',
    'django_dbbackup': 'django-dbbackup',
    'django_enumfields': 'django-enumfields',
    'django_environ': 'django-environ',
    'django_guardian': 'django-guardian',
    'django_haystack': 'django-haystack',
    'django_import_export': 'django-import-export',
    'django_js_asset': 'django-js-asset',
    'django_mptt': 'django-mptt',
    'django_oauth_toolkit': 'django-oauth-toolkit',
    'django_phonenumber_field': 'django-phonenumber-field',
    'django_polymorphic': 'django-polymorphic',
    'django_redis': 'django-redis',
    'django_rest_framework': 'djangorestframework',
    'django_reversion': 'django-reversion',
    'django_rosetta': 'django-rosetta',
    'django_simple_history': 'django-simple-history',
    'django_storages': 'django-storages',
    'django_tables2': 'django-tables2',
    'django_taggit': 'django-taggit',
    'django_treebeard': 'django-treebeard',
    'django_widget_tweaks': 'django-widget-tweaks',
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'gunicorn': 'gunicorn',
    'starlette': 'starlette',
    'pydantic': 'pydantic',
    'tornado': 'tornado',
    'sanic': 'sanic',
    'quart': 'Quart',
    'falcon': 'falcon',
    'bottle': 'bottle',
    'cherrypy': 'CherryPy',
    'pyramid': 'pyramid',
    'webob': 'WebOb',
    'werkzeug': 'Werkzeug',
    'jinja2': 'Jinja2',
    'mako': 'Mako',
    'twisted': 'Twisted',
    'scrapy': 'Scrapy',
    'selenium': 'selenium',
    'playwright': 'playwright',
    'mechanize': 'mechanize',
    'robobrowser': 'robobrowser',

    # ── Data Science & Numeric ──────────────────────────────────────────
    'numpy': 'numpy',
    'np': 'numpy',
    'pandas': 'pandas',
    'pd': 'pandas',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'scikit_learn': 'scikit-learn',
    'scikitlearn': 'scikit-learn',
    'statsmodels': 'statsmodels',
    'statmodels': 'statsmodels',
    'patsy': 'patsy',
    'sympy': 'sympy',
    'mpmath': 'mpmath',
    'numba': 'numba',
    'cupy': 'cupy',
    'jax': 'jax',
    'dask': 'dask',
    'ray': 'ray',
    'polars': 'polars',
    'vaex': 'vaex',
    'datatable': 'datatable',
    'pyspark': 'pyspark',
    'koalas': 'koalas',
    'modin': 'modin',

    # ── Machine Learning & AI ───────────────────────────────────────────
    'tensorflow': 'tensorflow',
    'tf': 'tensorflow',
    'torch': 'torch',
    'pytorch': 'torch',
    'keras': 'keras',
    'transformers': 'transformers',
    'datasets': 'datasets',
    'tokenizers': 'tokenizers',
    'accelerate': 'accelerate',
    'diffusers': 'diffusers',
    'sentence_transformers': 'sentence-transformers',
    'sentencepiece': 'sentencepiece',
    'tiktoken': 'tiktoken',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    'imblearn': 'imbalanced-learn',
    'imbalanced_learn': 'imbalanced-learn',
    'gensim': 'gensim',
    'nltk': 'nltk',
    'spacy': 'spacy',
    'textblob': 'textblob',
    'vaderSentiment': 'vaderSentiment',
    'stanza': 'stanza',
    'flair': 'flair',
    'fasttext': 'fasttext',
    'glove': 'glove-python-binary',
    'word2vec': 'gensim',
    'umap': 'umap-learn',
    'umap_learn': 'umap-learn',
    'hdbscan': 'hdbscan',
    'tsne': 'scikit-learn',
    'pca': 'scikit-learn',
    'lda': 'scikit-learn',
    'opencv': 'opencv-python',
    'cv2': 'opencv-python',
    'cv': 'opencv-python',
    'opencv_contrib': 'opencv-contrib-python',
    'opencv_headless': 'opencv-python-headless',
    'skimage': 'scikit-image',
    'scikit_image': 'scikit-image',
    'sklearn_crfsuite': 'sklearn-crfsuite',
    'eli5': 'eli5',
    'shap': 'shap',
    'lime': 'lime',
    'yellowbrick': 'yellowbrick',
    'mlflow': 'mlflow',
    'wandb': 'wandb',
    'optuna': 'optuna',
    'hyperopt': 'hyperopt',

    # ── Visualization ───────────────────────────────────────────────────
    'matplotlib': 'matplotlib',
    'mpl': 'matplotlib',
    'pyplot': 'matplotlib',
    'seaborn': 'seaborn',
    'sns': 'seaborn',
    'plotly': 'plotly',
    'plotly_express': 'plotly',
    'px': 'plotly',
    'bokeh': 'bokeh',
    'altair': 'altair',
    'ggplot': 'plotnine',
    'plotnine': 'plotnine',
    'holoviews': 'holoviews',
    'hvplot': 'hvplot',
    'datashader': 'datashader',
    'folium': 'folium',
    'geopandas': 'geopandas',
    'cartopy': 'Cartopy',
    'basemap': 'basemap',
    'pygal': 'pygal',
    'graphviz': 'graphviz',
    'pygraphviz': 'pygraphviz',
    'pydot': 'pydot',
    'networkx': 'networkx',
    'igraph': 'python-igraph',
    'pyvis': 'pyvis',
    'dash': 'dash',
    'streamlit': 'streamlit',
    'gradio': 'gradio',
    'panel': 'panel',
    'voila': 'voila',

    # ── Database ────────────────────────────────────────────────────────
    'sqlalchemy': 'SQLAlchemy',
    'sqlalchemy_utils': 'SQLAlchemy-Utils',
    'alembic': 'alembic',
    'psycopg2': 'psycopg2-binary',
    'psycopg': 'psycopg2-binary',
    'psycopg2cffi': 'psycopg2cffi',
    'asyncpg': 'asyncpg',
    'aiopg': 'aiopg',
    'aiosqlite': 'aiosqlite',
    'sqlite3': None,  # built-in
    'mysql': 'mysqlclient',
    'MySQLdb': 'mysqlclient',
    'mysql_connector': 'mysql-connector-python',
    'mysqlconnector': 'mysql-connector-python',
    'pymysql': 'PyMySQL',
    'mysqldb': 'mysqlclient',
    'pymongo': 'pymongo',
    'mongoengine': 'mongoengine',
    'motor': 'motor',
    'redis': 'redis',
    'rediscluster': 'redis-py-cluster',
    'aioredis': 'redis',
    'aredis': 'aredis',
    'elasticsearch': 'elasticsearch',
    'elasticsearch_dsl': 'elasticsearch-dsl',
    'opensearchpy': 'opensearch-py',
    'cassandra': 'cassandra-driver',
    'cassandra_driver': 'cassandra-driver',
    'neo4j': 'neo4j',
    'py2neo': 'py2neo',
    'influxdb': 'influxdb',
    'influxdb_client': 'influxdb-client',
    'kafka': 'kafka-python',
    'kafka_python': 'kafka-python',
    'confluent_kafka': 'confluent-kafka',
    'pika': 'pika',
    'celery': 'celery',
    'dramatiq': 'dramatiq',
    'rq': 'rq',
    'huey': 'huey',
    'fakeredis': 'fakeredis',
    'mongomock': 'mongomock',
    'testing_postgresql': 'testing.postgresql',
    'testing_mysqld': 'testing.mysqld',

    # ── Image & Media Processing ────────────────────────────────────────
    'PIL': 'Pillow',
    'Pillow': 'Pillow',
    'pil': 'Pillow',
    'pillow': 'Pillow',
    'imageio': 'imageio',
    'tifffile': 'tifffile',
    'pyexiv2': 'py3exiv2',
    'exifread': 'ExifRead',
    'piexif': 'piexif',
    'rawpy': 'rawpy',
    'pgmagick': 'pgmagick',
    'wand': 'Wand',
    'pyvips': 'pyvips',
    'moviepy': 'moviepy',
    'ffmpeg': 'ffmpeg-python',
    'ffmpeg_python': 'ffmpeg-python',
    'av': 'av',
    'pydub': 'pydub',
    'simpleaudio': 'simpleaudio',
    'pygame': 'pygame',
    'pyglet': 'pyglet',
    'arcade': 'arcade',
    'panda3d': 'panda3d',
    'cocos2d': 'cocos2d',
    'manim': 'manim',
    'manimlib': 'manimlib',

    # ── GUI & Desktop ───────────────────────────────────────────────────
    'tkinter': None,  # built-in
    'tk': None,  # built-in
    'Tkinter': None,  # built-in
    'ttk': None,  # built-in
    'PyQt5': 'PyQt5',
    'PyQt6': 'PyQt6',
    'PySide2': 'PySide2',
    'PySide6': 'PySide6',
    'PyQtGraph': 'pyqtgraph',
    'pyqtgraph': 'pyqtgraph',
    'wx': 'wxPython',
    'wxPython': 'wxPython',
    'kivy': 'kivy',
    'kivymd': 'kivymd',
    'flet': 'flet',
    'dearpygui': 'dearpygui',
    'tkinterweb': 'tkinterweb',
    'customtkinter': 'customtkinter',
    'tkinterdnd2': 'tkinterdnd2',
    'tkcalendar': 'tkcalendar',
    'ttkbootstrap': 'ttkbootstrap',
    'tkintermapview': 'tkintermapview',
    'CTk': 'customtkinter',
    'CTkMessagebox': 'CTkMessagebox',
    'CTkListbox': 'CTkListbox',
    'CTkTable': 'CTkTable',
    'CTkScrollableDropdown': 'CTkScrollableDropdown',

    # ── File Formats & Parsing ──────────────────────────────────────────
    'yaml': 'PyYAML',
    'pyyaml': 'PyYAML',
    'toml': 'toml',
    'tomli': 'tomli',
    'tomli_w': 'tomli-w',
    'json5': 'json5',
    'ujson': 'ujson',
    'orjson': 'orjson',
    'simplejson': 'simplejson',
    'rapidjson': 'python-rapidjson',
    'msgpack': 'msgpack',
    'msgpack_numpy': 'msgpack-numpy',
    'protobuf': 'protobuf',
    'avro': 'avro-python3',
    'parquet': 'pyarrow',
    'pyarrow': 'pyarrow',
    'fastparquet': 'fastparquet',
    'feather': 'pyarrow',
    'h5py': 'h5py',
    'tables': 'tables',
    'netCDF4': 'netCDF4',
    'xarray': 'xarray',
    'openpyxl': 'openpyxl',
    'xlrd': 'xlrd',
    'xlwt': 'xlwt',
    'xlsxwriter': 'XlsxWriter',
    'pyexcel': 'pyexcel',
    'pyexcel_xlsx': 'pyexcel-xlsx',
    'pyexcel_xls': 'pyexcel-xls',
    'pyexcel_ods': 'pyexcel-ods',
    'csv': None,  # built-in
    'csvkit': 'csvkit',
    'pandas_profiling': 'ydata-profiling',
    'ydata_profiling': 'ydata-profiling',
    'pdfplumber': 'pdfplumber',
    'PyPDF2': 'PyPDF2',
    'pypdf': 'pypdf',
    'pdfminer': 'pdfminer.six',
    'reportlab': 'reportlab',
    'fpdf': 'fpdf2',
    'docx': 'python-docx',
    'python_docx': 'python-docx',
    'pptx': 'python-pptx',
    'python_pptx': 'python-pptx',
    'odfpy': 'odfpy',

    # ── Cloud & DevOps ──────────────────────────────────────────────────
    'boto3': 'boto3',
    'botocore': 'botocore',
    'awscli': 'awscli',
    's3fs': 's3fs',
    'gcsfs': 'gcsfs',
    'adlfs': 'adlfs',
    'google_cloud': 'google-cloud',
    'google_cloud_storage': 'google-cloud-storage',
    'google_cloud_bigquery': 'google-cloud-bigquery',
    'google_cloud_pubsub': 'google-cloud-pubsub',
    'google_cloud_firestore': 'google-cloud-firestore',
    'google_cloud_vision': 'google-cloud-vision',
    'google_cloud_translate': 'google-cloud-translate',
    'google_cloud_speech': 'google-cloud-speech',
    'google_cloud_language': 'google-cloud-language',
    'google_cloud_automl': 'google-cloud-automl',
    'google_cloud_dlp': 'google-cloud-dlp',
    'google_cloud_tasks': 'google-cloud-tasks',
    'google_cloud_scheduler': 'google-cloud-scheduler',
    'google_cloud_functions': 'google-cloud-functions',
    'google_cloud_logging': 'google-cloud-logging',
    'google_cloud_monitoring': 'google-cloud-monitoring',
    'google_cloud_error_reporting': 'google-cloud-error-reporting',
    'google_cloud_trace': 'google-cloud-trace',
    'google_cloud_debugger': 'google-cloud-debugger',
    'google_cloud_profiler': 'google-cloud-profiler',
    'google_cloud_secret_manager': 'google-cloud-secret-manager',
    'google_cloud_kms': 'google-cloud-kms',
    'google_cloud_iam': 'google-cloud-iam',
    'google_cloud_resource_manager': 'google-cloud-resource-manager',
    'google_cloud_container': 'google-cloud-container',
    'google_cloud_dataproc': 'google-cloud-dataproc',
    'google_cloud_dataflow': 'google-cloud-dataflow',
    'google_cloud_datacatalog': 'google-cloud-datacatalog',
    'google_cloud_datalabeling': 'google-cloud-datalabeling',
    'google_cloud_dataplex': 'google-cloud-dataplex',
    'google_cloud_datastore': 'google-cloud-datastore',
    'google_cloud_spanner': 'google-cloud-spanner',
    'google_cloud_aiplatform': 'google-cloud-aiplatform',
    'google_cloud_notebooks': 'google-cloud-notebooks',
    'google_cloud_workflows': 'google-cloud-workflows',
    'google_cloud_billing': 'google-cloud-billing',
    'google_cloud_recommender': 'google-cloud-recommender',
    'google_cloud_org_policy': 'google-cloud-org-policy',
    'google_cloud_asset': 'google-cloud-asset',
    'google_cloud_securitycenter': 'google-cloud-securitycenter',
    'google_cloud_webrisk': 'google-cloud-webrisk',
    'google_cloud_websecurityscanner': 'google-cloud-websecurityscanner',
    'google_cloud_essential_contacts': 'google-cloud-essential-contacts',
    'google_cloud_network_connectivity': 'google-cloud-network-connectivity',
    'google_cloud_network_management': 'google-cloud-network-management',
    'google_cloud_network_services': 'google-cloud-network-services',
    'google_cloud_private_catalog': 'google-cloud-private-catalog',
    'google_cloud_service_directory': 'google-cloud-service-directory',
    'google_cloud_service_management': 'google-cloud-service-management',
    'google_cloud_service_usage': 'google-cloud-service-usage',
    'google_cloud_api_gateway': 'google-cloud-api-gateway',
    'google_cloud_endpoints': 'google-cloud-endpoints',
    'google_cloud_iap': 'google-cloud-iap',
    'google_cloud_ids': 'google-cloud-ids',
    'google_cloud_iot': 'google-cloud-iot',
    'google_cloud_media_translation': 'google-cloud-media-translation',
    'google_cloud_memcache': 'google-cloud-memcache',
    'google_cloud_redis': 'google-cloud-redis',
    'google_cloud_secret_manager': 'google-cloud-secret-manager',
    'google_cloud_security_center': 'google-cloud-security-center',
    'google_cloud_shell': 'google-cloud-shell',
    'google_cloud_source_context': 'google-cloud-source-context',
    'google_cloud_speech': 'google-cloud-speech',
    'google_cloud_talent': 'google-cloud-talent',
    'google_cloud_texttospeech': 'google-cloud-texttospeech',
    'google_cloud_tpu': 'google-cloud-tpu',
    'google_cloud_transcoder': 'google-cloud-transcoder',
    'google_cloud_videointelligence': 'google-cloud-videointelligence',
    'google_cloud_vision': 'google-cloud-vision',
    'google_cloud_vpc_access': 'google-cloud-vpc-access',
    'azure': 'azure',
    'azure_storage': 'azure-storage-blob',
    'azure_storage_blob': 'azure-storage-blob',
    'azure_storage_queue': 'azure-storage-queue',
    'azure_storage_file': 'azure-storage-file',
    'azure_storage_table': 'azure-storage-table',
    'azure_cosmos': 'azure-cosmos',
    'azure_servicebus': 'azure-servicebus',
    'azure_eventhub': 'azure-eventhub',
    'azure_identity': 'azure-identity',
    'azure_keyvault': 'azure-keyvault',
    'azure_mgmt': 'azure-mgmt',
    'azure_functions': 'azure-functions',
    'azure_durable_functions': 'azure-durable-functions',
    'azure_devops': 'azure-devops',
    'azure_cli': 'azure-cli',
    'azure_core': 'azure-core',
    'azure_common': 'azure-common',
    'azure_ai_formrecognizer': 'azure-ai-formrecognizer',
    'azure_ai_textanalytics': 'azure-ai-textanalytics',
    'azure_ai_vision': 'azure-ai-vision',
    'azure_ai_language': 'azure-ai-language',
    'azure_ai_translation': 'azure-ai-translation',
    'azure_ai_anomalydetector': 'azure-ai-anomalydetector',
    'azure_ai_personalizer': 'azure-ai-personalizer',
    'azure_ai_metricsadvisor': 'azure-ai-metricsadvisor',
    'azure_ai_contentsafety': 'azure-ai-contentsafety',
    'azure_ai_documentintelligence': 'azure-ai-documentintelligence',
    'azure_ai_inference': 'azure-ai-inference',
    'azure_ai_evaluation': 'azure-ai-evaluation',
    'azure_ai_projects': 'azure-ai-projects',
    'azure_ai_resources': 'azure-ai-resources',
    'kubernetes': 'kubernetes',
    'k8s': 'kubernetes',
    'docker': 'docker',
    'ansible': 'ansible',
    'ansible_runner': 'ansible-runner',
    'terraform': 'python-terraform',
    'pulumi': 'pulumi',
    'fabric': 'fabric',
    'paramiko': 'paramiko',
    'salt': 'salt',
    'testinfra': 'testinfra',
    'molecule': 'molecule',

    # ── Testing ─────────────────────────────────────────────────────────
    'pytest': 'pytest',
    'pytest_cov': 'pytest-cov',
    'pytest_xdist': 'pytest-xdist',
    'pytest_asyncio': 'pytest-asyncio',
    'pytest_mock': 'pytest-mock',
    'pytest_django': 'pytest-django',
    'pytest_flask': 'pytest-flask',
    'pytest_bdd': 'pytest-bdd',
    'pytest_benchmark': 'pytest-benchmark',
    'pytest_timeout': 'pytest-timeout',
    'pytest_rerunfailures': 'pytest-rerunfailures',
    'pytest_sugar': 'pytest-sugar',
    'pytest_html': 'pytest-html',
    'pytest_metadata': 'pytest-metadata',
    'pytest_ordering': 'pytest-ordering',
    'pytest_randomly': 'pytest-randomly',
    'pytest_repeat': 'pytest-repeat',
    'pytest_socket': 'pytest-socket',
    'pytest_subprocess': 'pytest-subprocess',
    'pytest_watch': 'pytest-watch',
    'pytest_freezegun': 'pytest-freezegun',
    'pytest_faker': 'pytest-faker',
    'pytest_lazyfixture': 'pytest-lazy-fixture',
    'unittest': None,  # built-in
    'unittest2': 'unittest2',
    'nose': 'nose',
    'nose2': 'nose2',
    'tox': 'tox',
    'nox': 'nox',
    'behave': 'behave',
    'lettuce': 'lettuce',
    'robot': 'robotframework',
    'robotframework': 'robotframework',
    'seleniumbase': 'seleniumbase',
    'playwright_pytest': 'pytest-playwright',
    'locust': 'locust',
    'hypothesis': 'hypothesis',
    'factory_boy': 'factory-boy',
    'faker': 'Faker',
    'freezegun': 'freezegun',
    'responses': 'responses',
    'httpretty': 'HTTPretty',
    'vcr': 'vcrpy',
    'vcrpy': 'vcrpy',
    'betamax': 'betamax',
    'mock': 'mock',
    'moto': 'moto',
    'coverage': 'coverage',

    # ── Security & Cryptography ─────────────────────────────────────────
    'cryptography': 'cryptography',
    'pycrypto': 'pycryptodome',
    'Crypto': 'pycryptodome',
    'pycryptodome': 'pycryptodome',
    'pycryptodomex': 'pycryptodomex',
    'bcrypt': 'bcrypt',
    'passlib': 'passlib',
    'jwt': 'PyJWT',
    'pyjwt': 'PyJWT',
    'oauthlib': 'oauthlib',
    'requests_oauthlib': 'requests-oauthlib',
    'authlib': 'Authlib',
    'python_jose': 'python-jose',
    'josepy': 'josepy',
    'acme': 'acme',
    'certbot': 'certbot',
    'letsencrypt': 'certbot',
    'paramiko': 'paramiko',
    'ssh': 'paramiko',
    'scp': 'scp',
    'pysftp': 'pysftp',
    'fabric': 'fabric',
    'invoke': 'invoke',
    'scapy': 'scapy',
    'pyshark': 'pyshark',
    'dpkt': 'dpkt',
    'impacket': 'impacket',
    'pcap': 'pypcap',
    'libpcap': 'pypcap',
    'nacl': 'PyNaCl',
    'pynacl': 'PyNaCl',
    'itsdangerous': 'itsdangerous',
    'blinker': 'blinker',
    'secretstorage': 'SecretStorage',
    'keyring': 'keyring',
    'keyrings_alt': 'keyrings.alt',

    # ── System & OS ─────────────────────────────────────────────────────
    'psutil': 'psutil',
    'distro': 'distro',
    'platformdirs': 'platformdirs',
    'appdirs': 'appdirs',
    'watchdog': 'watchdog',
    'schedule': 'schedule',
    'python_crontab': 'python-crontab',
    'daemonize': 'python-daemon',
    'daemon': 'python-daemon',
    'lockfile': 'lockfile',
    'pid': 'pid',
    'python_dotenv': 'python-dotenv',
    'dotenv': 'python-dotenv',
    'environs': 'environs',
    'dynaconf': 'dynaconf',
    'pydantic_settings': 'pydantic-settings',
    'omegaconf': 'omegaconf',
    'hydra': 'hydra-core',
    'config': 'configparser',  # built-in, but sometimes confused

    # ── CLI & Terminal ──────────────────────────────────────────────────
    'click': 'click',
    'typer': 'typer',
    'fire': 'fire',
    'argparse': None,  # built-in
    'docopt': 'docopt',
    'plac': 'plac',
    'cliff': 'cliff',
    'cement': 'cement',
    'python_fire': 'fire',
    'rich': 'rich',
    'rich_click': 'rich-click',
    'tqdm': 'tqdm',
    'progress': 'progress',
    'progressbar': 'progressbar2',
    'progressbar2': 'progressbar2',
    'alive_progress': 'alive-progress',
    'halo': 'halo',
    'yaspin': 'yaspin',
    'loguru': 'loguru',
    'colorlog': 'colorlog',
    'coloredlogs': 'coloredlogs',
    'termcolor': 'termcolor',
    'colorama': 'colorama',
    'blessings': 'blessings',
    'blessed': 'blessed',
    'textual': 'textual',
    'textual_dev': 'textual-dev',
    'prompt_toolkit': 'prompt-toolkit',
    'questionary': 'questionary',
    'pyinquirer': 'PyInquirer',
    'inquirer': 'inquirer',
    'python_terraform': 'python-terraform',

    # ── Text Processing ─────────────────────────────────────────────────
    're': None,  # built-in
    'regex': 'regex',
    'fuzzywuzzy': 'fuzzywuzzy',
    'levenshtein': 'python-Levenshtein',
    'python_levenshtein': 'python-Levenshtein',
    'rapidfuzz': 'rapidfuzz',
    'difflib': None,  # built-in
    'textdistance': 'textdistance',
    'jellyfish': 'jellyfish',
    'phonenumbers': 'phonenumbers',
    'python_whois': 'python-whois',
    'whois': 'python-whois',
    'user_agents': 'user-agents',
    'langdetect': 'langdetect',
    'langid': 'langid',
    'polyglot': 'polyglot',
    'translate': 'translate',
    'googletrans': 'googletrans',
    'pycountry': 'pycountry',
    'currencyconverter': 'CurrencyConverter',
    'num2words': 'num2words',
    'inflect': 'inflect',
    'inflection': 'inflection',
    'pluralize': 'inflect',
    'slugify': 'python-slugify',
    'python_slugify': 'python-slugify',
    'unidecode': 'Unidecode',
    'ftfy': 'ftfy',
    'emoji': 'emoji',
    'markdown': 'markdown',
    'markdown2': 'markdown2',
    'mistune': 'mistune',
    'commonmark': 'commonmark',
    'pandoc': 'pandoc',

    # ── Date & Time ─────────────────────────────────────────────────────
    'dateutil': 'python-dateutil',
    'python_dateutil': 'python-dateutil',
    'arrow': 'arrow',
    'pendulum': 'pendulum',
    'pytz': 'pytz',
    'tzdata': 'tzdata',
    'tzlocal': 'tzlocal',
    'moment': 'moment',
    'maya': 'maya',
    'delorean': 'Delorean',
    'whenever': 'whenever',
    'dateparser': 'dateparser',
    'parsedatetime': 'parsedatetime',
    'relativedelta': 'python-dateutil',
    'isodate': 'isodate',
    'aniso8601': 'aniso8601',
    'ciso8601': 'ciso8601',
    'timeago': 'timeago',

    # ── Concurrency & Parallelism ───────────────────────────────────────
    'concurrent_futures': None,  # built-in
    'multiprocessing': None,  # built-in
    'threading': None,  # built-in
    'asyncio': None,  # built-in
    'gevent': 'gevent',
    'eventlet': 'eventlet',
    'greenlet': 'greenlet',
    'trio': 'trio',
    'anyio': 'anyio',
    'curio': 'curio',
    'uvloop': 'uvloop',

    # ── Serialization ───────────────────────────────────────────────────
    'pickle': None,  # built-in
    'json': None,  # built-in
    'marshmallow': 'marshmallow',
    'marshmallow_dataclass': 'marshmallow-dataclass',
    'marshmallow_enum': 'marshmallow-enum',
    'marshmallow_oneofschema': 'marshmallow-oneofschema',
    'marshmallow_sqlalchemy': 'marshmallow-sqlalchemy',
    'marshmallow_jsonapi': 'marshmallow-jsonapi',
    'pydantic': 'pydantic',
    'attrs': 'attrs',
    'dataclasses': None,  # built-in
    'cattrs': 'cattrs',
    'serde': 'pyserde',
    'dataclasses_json': 'dataclasses-json',
    'mashumaro': 'mashumaro',

    # ── Scientific & Engineering ────────────────────────────────────────
    'astropy': 'astropy',
    'sunpy': 'sunpy',
    'biopython': 'biopython',
    'Bio': 'biopython',
    'obspy': 'obspy',
    'pint': 'pint',
    'quantities': 'quantities',
    'uncertainties': 'uncertainties',
    'lmfit': 'lmfit',
    'emcee': 'emcee',
    'corner': 'corner',
    'dynesty': 'dynesty',
    'pymc': 'pymc',
    'pymc3': 'pymc3',
    'arviz': 'arviz',
    'bambi': 'bambi',
    'prophet': 'prophet',
    'fbprophet': 'fbprophet',
    'pmdarima': 'pmdarima',
    'tsfresh': 'tsfresh',
    'sktime': 'sktime',
    'pyod': 'pyod',

    # ── API Clients & Wrappers ──────────────────────────────────────────
    'twilio': 'twilio',
    'stripe': 'stripe',
    'paypal': 'paypalrestsdk',
    'square': 'squareup',
    'shopify': 'shopifyapi',
    'github': 'PyGithub',
    'pygithub': 'PyGithub',
    'gitlab': 'python-gitlab',
    'python_gitlab': 'python-gitlab',
    'bitbucket': 'bitbucket-api',
    'jira': 'jira',
    'confluence': 'atlassian-python-api',
    'atlassian': 'atlassian-python-api',
    'slack': 'slack-sdk',
    'slack_sdk': 'slack-sdk',
    'slacker': 'slacker',
    'discord': 'discord.py',
    'discordpy': 'discord.py',
    'telegram': 'python-telegram-bot',
    'telegram_bot': 'python-telegram-bot',
    'telebot': 'pyTelegramBotAPI',
    'pytelegrambotapi': 'pyTelegramBotAPI',
    'whatsapp': 'twilio',
    'twitter': 'tweepy',
    'tweepy': 'tweepy',
    'linkedin': 'python-linkedin',
    'reddit': 'praw',
    'praw': 'praw',
    'spotify': 'spotipy',
    'spotipy': 'spotipy',
    'youtube': 'google-api-python-client',
    'googleapiclient': 'google-api-python-client',
    'google_api_python_client': 'google-api-python-client',
    'gspread': 'gspread',
    'oauth2client': 'oauth2client',
    'google_auth': 'google-auth',
    'google_auth_oauthlib': 'google-auth-oauthlib',
    'google_auth_httplib2': 'google-auth-httplib2',

    # ── Development Tools ───────────────────────────────────────────────
    'black': 'black',
    'ruff': 'ruff',
    'flake8': 'flake8',
    'pylint': 'pylint',
    'mypy': 'mypy',
    'isort': 'isort',
    'pre_commit': 'pre-commit',
    'commitizen': 'commitizen',
    'poetry': 'poetry',
    'pipenv': 'pipenv',
    'virtualenv': 'virtualenv',
    'venv': None,  # built-in
    'pyenv': 'pyenv',
    'pip': 'pip',
    'setuptools': 'setuptools',
    'wheel': 'wheel',
    'twine': 'twine',
    'build': 'build',
    'hatch': 'hatch',
    'pdm': 'pdm',
    'bumpver': 'bumpver',
    'bumpversion': 'bumpversion',
    'invoke': 'invoke',
    'doit': 'doit',
    'snakeviz': 'snakeviz',
    'pyinstrument': 'pyinstrument',
    'line_profiler': 'line-profiler',
    'memory_profiler': 'memory-profiler',
    'py_spy': 'py-spy',
    'pdb': None,  # built-in
    'ipdb': 'ipdb',
    'pdbpp': 'pdbpp',
    'pudb': 'pudb',
    'wdb': 'wdb',
    'debugpy': 'debugpy',
    'ptvsd': 'ptvsd',

    # ── Jupyter & Notebooks ─────────────────────────────────────────────
    'jupyter': 'jupyter',
    'notebook': 'notebook',
    'jupyterlab': 'jupyterlab',
    'ipython': 'ipython',
    'ipykernel': 'ipykernel',
    'nbformat': 'nbformat',
    'nbconvert': 'nbconvert',
    'nbval': 'nbval',
    'papermill': 'papermill',
    'scrapbook': 'scrapbook',
    'jupytext': 'jupytext',
    'rise': 'RISE',
    'nbdime': 'nbdime',
    'nbviewer': 'nbviewer',

    # ── Utility ─────────────────────────────────────────────────────────
    'retry': 'retry',
    'tenacity': 'tenacity',
    'backoff': 'backoff',
    'retrying': 'retrying',
    'cachetools': 'cachetools',
    'diskcache': 'diskcache',
    'joblib': 'joblib',
    'cloudpickle': 'cloudpickle',
    'dill': 'dill',
    'pickle5': 'pickle5',
    'blosc': 'blosc',
    'zstandard': 'zstandard',
    'lz4': 'lz4',
    'python_snappy': 'python-snappy',
    'snappy': 'python-snappy',
    'brotli': 'Brotli',
    'brotlicffi': 'brotlicffi',
    'zopfli': 'zopfli',
    'pyzstd': 'pyzstd',
    'python_magic': 'python-magic',
    'magic': 'python-magic',
    'mimetypes': None,  # built-in
    'pathspec': 'pathspec',
    'glob2': 'glob2',
    'wcmatch': 'wcmatch',
    'send2trash': 'Send2Trash',
    'pyperclip': 'pyperclip',
    'clipboard': 'clipboard',
    'notify': 'notify2',
    'notify2': 'notify2',
    'desktop_notifier': 'desktop-notifier',
    'plyer': 'plyer',
    'pynput': 'pynput',
    'keyboard': 'keyboard',
    'mouse': 'mouse',
    'pyautogui': 'PyAutoGUI',
    'pygetwindow': 'PyGetWindow',
    'pyrect': 'PyRect',
    'pyscreeze': 'PyScreeze',
    'pytweening': 'pytweening',
    'pymsgbox': 'PyMsgBox',
    'opencv_python': 'opencv-python',
}


# ===========================================================================
# Standard library names (cross-version)
# ===========================================================================

def get_stdlib_names() -> FrozenSet[str]:
    """
    Return a frozenset of stdlib top-level module names.

    Uses ``sys.stdlib_module_names`` (Python 3.10+) if available.
    Falls back to a hardcoded comprehensive list for older versions.

    Returns
    -------
    frozenset of str
    """
    if hasattr(sys, 'stdlib_module_names'):
        return frozenset(sys.stdlib_module_names)

    return frozenset({
        '__future__', '__main__', '_thread', '_dummy_thread',
        'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
        'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii',
        'binhex', 'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb',
        'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
        'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
        'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv',
        'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal',
        'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings',
        'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
        'fnmatch', 'formatter', 'fractions', 'ftplib', 'functools', 'gc',
        'getopt', 'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip',
        'hashlib', 'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib',
        'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress',
        'itertools', 'json', 'keyword', 'lib2to3', 'linecache', 'locale',
        'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math',
        'mimetypes', 'mmap', 'modulefinder', 'multiprocessing', 'netrc',
        'nis', 'nntplib', 'numbers', 'operator', 'optparse', 'os',
        'ossaudiodev', 'parser', 'pathlib', 'pdb', 'pickle', 'pickletools',
        'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix',
        'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'pwd',
        'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random',
        're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy',
        'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex',
        'shutil', 'signal', 'site', 'smtpd', 'smtplib', 'sndhdr',
        'socket', 'socketserver', 'sqlite3', 'ssl', 'stat', 'statistics',
        'string', 'stringprep', 'struct', 'subprocess', 'sunau', 'symtable',
        'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile', 'telnetlib',
        'tempfile', 'termios', 'test', 'textwrap', 'threading', 'time',
        'timeit', 'tkinter', 'token', 'tokenize', 'trace', 'traceback',
        'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing',
        'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv',
        'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound',
        'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile',
        'zipimport', 'zlib', 'zoneinfo',
    })


# ===========================================================================
# System / platform-internal modules (never pip-installable)
# ===========================================================================

_SYSTEM_MODULE_PATTERNS: tuple = (
    lambda name: name.startswith('_'),
    lambda name: len(name) <= 3 and name.islower() and name not in (
        'pip', 'uv', 'tqdm', 'jwt', 'bs4', 'cv2', 'PIL', 'wx', 'toml',
        'httpx', 'rio', 'ray', 'jax', 'tf', 'np', 'pd', 'mpl', 'sns',
        'px', 'Bio', 'io', 'os',
    ),
    lambda name: name in (
        'nt', 'posix', 'ntpath', 'posixpath', 'genericpath',
        'os2', 'mac', 'ce', 'riscos', 'vms', 'win32api', 'win32con',
        'win32com', 'win32file', 'win32gui', 'win32process',
        'pywintypes', 'pythoncom',
    ),
)


# ===========================================================================
# Public API
# ===========================================================================

def resolve_pip_name(import_name: str) -> str:
    """
    Convert an import name to its pip-installable package name.

    Checks the comprehensive mapping first. Falls back to normalizing
    the name (lowercase, hyphens instead of underscores).

    Parameters
    ----------
    import_name : str
        The top-level import name (e.g., ``'sklearn'``).

    Returns
    -------
    str
        The pip package name (e.g., ``'scikit-learn'``).
    """
    top_level = import_name.split('.')[0]
    if top_level in _PIP_NAME_MAP:
        return _PIP_NAME_MAP[top_level]
    return top_level.lower().replace('_', '-')


def is_builtin(name: str) -> bool:
    """
    Check if a module name is a built-in C module.

    Parameters
    ----------
    name : str
        The top-level module name.

    Returns
    -------
    bool
    """
    return name in sys.builtin_module_names


def is_stdlib(name: str) -> bool:
    """
    Check if a module name is in the standard library.

    Parameters
    ----------
    name : str
        The top-level module name.

    Returns
    -------
    bool
    """
    return name in get_stdlib_names()


def is_system_module(name: str) -> bool:
    """
    Check if a module name matches system/internal patterns.

    Parameters
    ----------
    name : str
        The top-level module name.

    Returns
    -------
    bool
    """
    for pattern in _SYSTEM_MODULE_PATTERNS:
        try:
            if pattern(name):
                return True
        except Exception:
            continue
    return False


def is_already_importable(name: str) -> bool:
    """
    Check if a module can already be imported (installed or stdlib).

    Uses ``importlib.util.find_spec`` for a lightweight check that
    does not actually execute the module code.

    Parameters
    ----------
    name : str
        The full module name (may include dots for submodules).

    Returns
    -------
    bool
    """
    return importlib.util.find_spec(name) is not None


def should_skip_install(
    name: str,
    failed_packages: Set[str],
    installing: Set[str],
) -> bool:
    """
    Determine whether an import should bypass auto-installation.

    Parameters
    ----------
    name : str
        The full import name (may include dots).
    failed_packages : set of str
        Packages that previously failed installation.
    installing : set of str
        Packages currently being installed.

    Returns
    -------
    bool
        True if installation should be skipped.
    """
    # Relative imports
    if name.startswith('.'):
        return True

    top_level = name.split('.')[0]

    # Built-in modules
    if is_builtin(top_level):
        return True

    # Stdlib modules
    if is_stdlib(top_level):
        return True

    # System modules
    if is_system_module(top_level):
        return True

    # Submodule of an already-installed package
    if '.' in name:
        if is_already_importable(top_level):
            return True

    # Previously failed
    if name in failed_packages or top_level in failed_packages:
        return True

    # Currently installing
    if name in installing or top_level in installing:
        return True

    # Path-like or empty names
    if not name or any(c in name for c in ('/', '\\', ' ')):
        return True

    return False


def add_pip_mapping(import_name: str, pip_name: str) -> None:
    """
    Register a custom import-to-pip name mapping at runtime.

    Parameters
    ----------
    import_name : str
        The import name (e.g., ``'mylib'``).
    pip_name : str
        The pip package name (e.g., ``'my-custom-lib'``).
    """
    _PIP_NAME_MAP[import_name] = pip_name