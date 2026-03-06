import unittest
from pydantic import ValidationError
from mem0.configs.vector_stores.redis import RedisDBConfig
from mem0.configs.vector_stores.qdrant import QdrantConfig
from mem0.configs.vector_stores.azure_ai_search import AzureAISearchConfig

class TestVectorStoreConfigs(unittest.TestCase):
    def test_redis_config_extra_fields(self):
        # Valid config
        config = RedisDBConfig(redis_url="redis://localhost:6379")
        self.assertEqual(config.redis_url, "redis://localhost:6379")

        # Extra fields should raise ValidationError
        with self.assertRaises(ValidationError) as cm:
            RedisDBConfig(redis_url="redis://localhost:6379", extra_field="invalid")
        self.assertIn("extra_field", str(cm.exception))
        self.assertIn("Extra inputs are not permitted", str(cm.exception))

    def test_qdrant_config_extra_fields(self):
        # Valid config
        config = QdrantConfig(path="/tmp/qdrant")
        self.assertEqual(config.path, "/tmp/qdrant")

        # Extra fields should raise ValidationError
        with self.assertRaises(ValidationError) as cm:
            QdrantConfig(path="/tmp/qdrant", extra_field="invalid")
        self.assertIn("extra_field", str(cm.exception))
        self.assertIn("Extra inputs are not permitted", str(cm.exception))

    def test_azure_ai_search_config_helpful_error(self):
        # Test the custom helpful error message for use_compression
        with self.assertRaises(ValidationError) as cm:
            AzureAISearchConfig(
                service_name="test",
                api_key="test",
                use_compression=True
            )
        self.assertIn("The parameter 'use_compression' is no longer supported", str(cm.exception))

        # Test other extra fields
        with self.assertRaises(ValidationError) as cm:
            AzureAISearchConfig(
                service_name="test",
                api_key="test",
                extra_field="invalid"
            )
        self.assertIn("extra_field", str(cm.exception))
        self.assertIn("Extra inputs are not permitted", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
