import io
import json
import os
import unittest
from unittest.mock import patch

import app_llm


class TestLlmProviderFile(unittest.TestCase):
    def _load_text(self, text):
        with patch.dict(
            os.environ,
            {'LLM_PROVIDERS_FILE': '/private/providers.json'},
            clear=False,
        ), patch('builtins.open', return_value=io.StringIO(text)):
            return app_llm._load_providers()

    def _load(self, payload):
        return self._load_text(json.dumps(payload))

    def test_loads_private_provider_file(self):
        providers = self._load({
            'providers': [{
                'id': 'proxy',
                'name': 'Proxy',
                'base_url': 'https://example.test/v1',
                'api_key': 'secret-key',
                'models': ['model-a', 'model-b'],
                'default_model': 'model-b',
            }],
        })

        self.assertEqual(list(providers), ['proxy'])
        self.assertEqual(providers['proxy']['api_key'], 'secret-key')
        self.assertEqual(providers['proxy']['default_model'], 'model-b')

    def test_filters_incomplete_providers(self):
        providers = self._load({'providers': [
            {'id': 'no-key', 'base_url': 'https://example.test/v1', 'models': ['m']},
            {'id': 'no-model', 'base_url': 'https://example.test/v1', 'api_key': 'k'},
        ]})

        self.assertEqual(providers, {})

    def test_accepts_top_level_list(self):
        providers = self._load([{
            'id': 'proxy',
            'base_url': 'https://example.test/v1',
            'api_key': 'secret-key',
            'models': ['model-a'],
        }])

        self.assertEqual(providers['proxy']['default_model'], 'model-a')

    def test_malformed_file_returns_no_providers(self):
        self.assertEqual(self._load_text('{not-json'), {})


if __name__ == '__main__':
    unittest.main()
