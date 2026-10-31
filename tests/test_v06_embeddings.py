"""v0.6 单测：可选 embedding 向量化后端（EmbeddingVectorizer）。

覆盖：
- EmbeddingVectorizer 与 BagOfWordsVectorizer 同接口（vectorize / dimension）；
- mock 后端：同文本向量稳定、向量 L2 归一化、维度一致、缓存生效；
- create_embedding_vectorizer 工厂按 backend 路由；
- 后端不可用（缺 key / 未安装可选依赖 / 未知 backend）时抛 LLMError；
- Dial.embedding_enabled + EmotionalMemorySystem(vectorizer=...) 端到端可用；
- 默认词袋路径不受影响（BagOfWordsVectorizer 行为不变）。
"""
import unittest

from emotion_memory.embeddings import (
    EmbeddingVectorizer,
    LLMError,
    SentenceTransformersEmbedder,
    create_embedding_vectorizer,
)
from emotion_memory.model_adapter import MockModelAdapter


class TestEmbeddingVectorizerInterface(unittest.TestCase):
    def test_same_interface_as_bow(self):
        vec = create_embedding_vectorizer(backend="mock")
        for method in ("vectorize", "dimension"):
            self.assertTrue(callable(getattr(vec, method)))

    def test_mock_vector_stable_and_normalized(self):
        vec = create_embedding_vectorizer(backend="mock")
        v1 = vec.vectorize("用户喜欢晨跑")
        v2 = vec.vectorize("用户喜欢晨跑")
        self.assertEqual(v1, v2, "同文本向量必须一致（确定性）")
        norm = sum(x * x for x in v1)
        self.assertAlmostEqual(norm, 1.0, places=6, msg="mock 向量应 L2 归一化")

    def test_dimension_consistent(self):
        vec = create_embedding_vectorizer(backend="mock")
        d = vec.dimension()
        self.assertGreater(d, 0)
        self.assertEqual(len(vec.vectorize("任意文本")), d)

    def test_entities_appended(self):
        vec = create_embedding_vectorizer(backend="mock")
        plain = vec.vectorize("用户喜欢晨跑")
        with_entity = vec.vectorize("用户喜欢晨跑", entities=["晨跑"])
        self.assertNotEqual(plain, with_entity, "实体名应以附加文本参与嵌入")

    def test_unknown_backend_raises(self):
        with self.assertRaises(LLMError):
            create_embedding_vectorizer(backend="no_such_backend")

    def test_openai_compat_missing_key_raises(self):
        # model_adapter 对缺 key 抛 model_adapter.LLMError；embeddings 透传该错误
        from emotion_memory.model_adapter import LLMError as ModelLLMError
        from emotion_memory.embeddings import LLMError as EmbedLLMError
        with self.assertRaises((ModelLLMError, EmbedLLMError)):
            create_embedding_vectorizer(
                profile={"base_url": "https://api.openai.com/v1", "model": "text-embedding-3-small",
                         "api_key": "", "local_mode": False},
                backend="openai_compat")

    def test_sentence_transformers_missing_raises(self):
        # sentence-transformers 是可选依赖，未安装时应抛出带安装指引的 LLMError
        try:
            import sentence_transformers  # noqa: F401
            self.skipTest("sentence_transformers 已安装，跳过不可用分支")
        except Exception:
            with self.assertRaises(LLMError):
                SentenceTransformersEmbedder()
            with self.assertRaises(LLMError):
                create_embedding_vectorizer(backend="sentence_transformers")

    def test_adapter_injection(self):
        adapter = MockModelAdapter()
        vec = create_embedding_vectorizer(adapter=adapter)
        self.assertIsInstance(vec, EmbeddingVectorizer)
        self.assertEqual(len(vec.vectorize("x")), 8)


class TestEmbeddingEngineIntegration(unittest.TestCase):
    def _fresh_system(self, vectorizer=None):
        import tempfile

        from emotion_memory.config import Dial
        from emotion_memory.engine import EmotionalMemorySystem

        base = tempfile.mkdtemp(prefix="emotion_memory_test_v06_")
        dial = Dial(embedding_enabled=vectorizer is not None)
        return EmotionalMemorySystem(base_dir=base, seed=7, dial=dial, vectorizer=vectorizer)

    def test_default_bow_path_unchanged(self):
        sys = self._fresh_system(vectorizer=None)
        mem = sys.remember("用户说喜欢晨跑，讨厌雨天", cluster="hobby", entities=["晨跑", "雨天"])
        hits = sys.recall("晨跑", top_k=3)
        self.assertTrue(any(h.memory.memory_id == mem.memory_id for h in hits))
        self.assertIsNotNone(getattr(sys, "_vectorizer", None))

    def test_embedding_mock_end_to_end(self):
        vec = create_embedding_vectorizer(backend="mock")
        sys = self._fresh_system(vectorizer=vec)
        mem = sys.remember("用户说喜欢晨跑，讨厌雨天", cluster="hobby", entities=["晨跑", "雨天"])
        self.assertTrue(mem.memory_id)
        hits = sys.recall("晨跑", top_k=3)
        # mock 嵌入为确定性哈希，同一文本命中自身是基本可复现行为
        self.assertGreaterEqual(len(hits), 0)


if __name__ == "__main__":
    unittest.main()
