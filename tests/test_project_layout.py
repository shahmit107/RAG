"""Check relocated entry points without models, API keys, or persistent writes."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]

# Each check runs in its own process to isolate import caches and module globals.
BOOTSTRAP = r'''
import importlib
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

def stub(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module

collection = MagicMock()
collection.count.return_value = 1
collection.get.return_value = {
    'documents': ['sample document'],
    'metadatas': [{'source': 'sample.txt', 'page': -1}],
    'ids': ['sample'],
}
client = MagicMock()
client.get_or_create_collection.return_value = collection
client.get_collection.return_value = collection
client.list_collections.return_value = [SimpleNamespace(name='my_documents')]
database_factory = MagicMock(return_value=client)
stub('chromadb', PersistentClient=database_factory)

# Skip environment-file reads and every remote provider.
dotenv_loader = MagicMock(return_value=True)
stub('dotenv', load_dotenv=dotenv_loader)
provider_client = MagicMock()
provider_client.models.embed_content.side_effect = lambda **kw: SimpleNamespace(
    embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in (
        kw['contents'] if isinstance(kw['contents'], list) else [kw['contents']]
    )]
)
genai = stub('google.genai', Client=MagicMock(return_value=provider_client))
stub('google', genai=genai)
stub('groq', Groq=MagicMock())
stub('sentence_transformers', CrossEncoder=MagicMock())
stub('rank_bm25', BM25Okapi=MagicMock())

document = SimpleNamespace(page_content='sample document', metadata={'source': 'sample.txt'})
loader = MagicMock()
loader.load.return_value = [document]
loader_factory = MagicMock(return_value=loader)
stub('langchain_community')
stub('langchain_community.document_loaders',
     DirectoryLoader=loader_factory, TextLoader=loader_factory, PyMuPDFLoader=loader_factory)
splitter = MagicMock()
splitter.split_documents.side_effect = lambda documents: list(documents)
stub('langchain_text_splitters', RecursiveCharacterTextSplitter=MagicMock(return_value=splitter))
stub('tiktoken', get_encoding=MagicMock(return_value=SimpleNamespace(encode=lambda text: list(text))))

ui = stub('streamlit')
for name in ('set_page_config', 'markdown', 'title', 'caption', 'success', 'warning', 'error'):
    setattr(ui, name, MagicMock())
ui.file_uploader = MagicMock(return_value=None)
ui.button = MagicMock(return_value=False)
ui.text_input = MagicMock(return_value='')
ui.session_state = {}
stub('requests')
'''


class ProjectLayoutTests(unittest.TestCase):
    def run_isolated(self, code):
        source = f"import sys; sys.path.insert(0, {str(ROOT)!r})\n" + BOOTSTRAP + code
        environment = os.environ.copy()
        environment['GEMINI_API_KEY'] = 'layout-test-placeholder'
        environment['GROQ_API_KEY'] = 'layout-test-placeholder'
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='.layout-') as directory:
            assert Path(directory).resolve().is_relative_to(ROOT / 'tests')
            result = subprocess.run(
                [sys.executable, '-c', source], cwd=directory, env=environment,
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_project_imports_and_syntax(self):
        files = list(ROOT.glob('*.py'))
        for directory in ('backend', 'frontend', 'scripts', 'tests'):
            files.extend((ROOT / directory).rglob('*.py'))
        old_modules = {'chroma_db', 'chunking', 'embedding_manager', 'retrieval', 'generation', 'query'}
        for path in files:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn(node.module, old_modules, str(path))
                    if node.module.startswith(('backend.', 'scripts.', 'frontend.')):
                        module_path = ROOT.joinpath(*node.module.split('.'))
                        self.assertTrue(
                            module_path.with_suffix('.py').exists()
                            or (module_path / '__init__.py').exists(), str(path),
                        )

    def test_data_paths_and_environment_are_independent_of_working_directory(self):
        self.run_isolated('''
from backend.paths import PROJECT_ROOT, DATABASE_DIR, DOCUMENTS_DIR, UPLOADS_DIR, ENV_PATH
from backend.pipeline import chroma_db, chunking, embedding_manager, query
from scripts import print_db
assert PROJECT_ROOT == Path(sys.path[0])
assert Path(chunking.FOLDER_PATH) == DOCUMENTS_DIR == PROJECT_ROOT / 'Pdfs'
assert Path(print_db.DB_PATH) == DATABASE_DIR == PROJECT_ROOT / 'my_local_db'
database_factory.assert_called_once_with(path=str(DATABASE_DIR))
assert UPLOADS_DIR == PROJECT_ROOT / 'storage' / 'documents'
assert ENV_PATH == PROJECT_ROOT / '.env'
assert all(call.args == (ENV_PATH,) for call in dotenv_loader.call_args_list)
''')

    def test_server_launcher_preserves_routes_and_shared_state(self):
        self.run_isolated('''
import server
from backend import server as implementation
from backend.pipeline import chroma_db
assert server is implementation
assert server.collection is chroma_db.collection
assert {route.path for route in server.app.routes} >= {'/upload', '/query'}
pipeline = SimpleNamespace(run=MagicMock(return_value='sample answer'))
server.rag = pipeline
assert implementation.rag is pipeline
assert server.query_rag(' question ') == {'answer': 'sample answer'}
pipeline.run.assert_called_once_with('question')
runner = MagicMock()
stub('uvicorn', run=runner)
runpy.run_path(str(Path(sys.path[0]) / 'server.py'), run_name='__main__')
runner.assert_called_once_with(server.app, host='127.0.0.1', port=8000)
''')

    def test_upload_keeps_response_shape_and_uses_relocated_pipeline(self):
        self.run_isolated('''
import server
from backend.paths import UPLOADS_DIR
assert server.STORAGE_DIR == UPLOADS_DIR
# Redirect this check's uploads away from the real documents.
server.STORAGE_DIR = Path.cwd() / 'documents'
response = server.upload_file('../example.txt', b'sample document')
assert set(response) == {'document_id', 'filename', 'chunks'}
assert response['filename'] == 'example.txt' and response['chunks'] == 1
stored = server.STORAGE_DIR / (response['document_id'] + '.txt')
assert stored.read_bytes() == b'sample document'
call = collection.add.call_args.kwargs
assert call['documents'] == ['sample document']
assert call['metadatas'] == [{'source': 'example.txt', 'page': -1}]
assert call['ids'] == ['upload_' + response['document_id'] + '_0']
''')

    def test_ingestion_launcher_uses_existing_document_and_database_paths(self):
        self.run_isolated('''
from backend.paths import ENV_PATH, DOCUMENTS_DIR, DATABASE_DIR
runpy.run_path(str(Path(sys.path[0]) / 'main.py'), run_name='__main__')
dotenv_loader.assert_any_call(ENV_PATH, override=True)
assert len(loader_factory.call_args_list) == 2
text_call, pdf_call = loader_factory.call_args_list
assert text_call.kwargs['path'] == str(DOCUMENTS_DIR)
assert pdf_call.args[0] == str(DOCUMENTS_DIR)
database_factory.assert_called_once_with(path=str(DATABASE_DIR))
assert collection.add.call_args.kwargs['ids'] == ['chunk_0', 'chunk_1']
''')

    def test_database_inspection_launcher(self):
        self.run_isolated('''
import print_db
from scripts import print_db as implementation
from backend.paths import DATABASE_DIR
assert print_db is implementation
assert Path(print_db.DB_PATH) == DATABASE_DIR
runpy.run_path(str(Path(sys.path[0]) / 'print_db.py'), run_name='__main__')
database_factory.assert_called_once_with(path=str(DATABASE_DIR))
client.get_collection.assert_called_once_with('my_documents')
collection.get.assert_called_once_with(include=['documents', 'metadatas'])
''')

    def test_streamlit_launcher_runs_existing_interface(self):
        self.run_isolated('''
runpy.run_path(str(Path(sys.path[0]) / 'streamlit_app.py'), run_name='__main__')
ui.set_page_config.assert_called_once()
assert ui.set_page_config.call_args.kwargs['page_title'] == 'RAG Q&A'
ui.file_uploader.assert_called_once_with('Document', type=['pdf', 'txt'], max_upload_size=25)
assert ui.button.call_count == 2
''')


if __name__ == '__main__':
    unittest.main()
