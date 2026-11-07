"""v0.6.3 单测：本地 ONNX embedding 后端（FastEmbedder）与「改写查询」对照。

覆盖：
- FastEmbedder / fastembed 后端缺依赖时抛带安装指引的 LLMError；
- ``auto`` 后端回退链：sentence_transformers 不可用时回退 fastembed，
  两者都不可用时聚合报错；
- ``_default_fastembed_cache`` 的缓存目录解析（环境变量优先、Windows 用盘根短路径，
  规避 WinError 206 超长路径）；
- benchmark 改写查询集：覆盖全部样例、且与原文不复用 3 字以上字面片段
  （保证词面重合优势不污染对照）；
- 默认词袋路径不受影响。
"""
import os
import unittest
from unittest import mock

from emotion_memory.embeddings import (
    EmbeddingVectorizer,
    FastEmbedder,
    LLMError,
    _default_fastembed_cache,
    create_embedding_vectorizer,
)


class TestFastEmbedderAvailability(unittest.TestCase):
    def test_missing_dependency_raises_with_hint(self):
        """fastembed 未安装时应抛 LLMError 并给出安装指引。"""
        with mock.patch.dict("sys.modules", {"fastembed": None}):
            with self.assertRaises(LLMError) as ctx:
                FastEmbedder()
        self.assertIn("fastembed", str(ctx.exception))
        self.assertIn("pip install", str(ctx.exception))

    def test_constructor_failure_wrapped_as_llm_error(self):
        """模型加载失败（网络/缓存异常）应包装为 LLMError，不泄漏原始异常类型。"""
        fake_mod = mock.MagicMock()
        fake_mod.TextEmbedding.side_effect = RuntimeError("boom")
        with mock.patch.dict("sys.modules", {"fastembed": fake_mod}):
            with self.assertRaises(LLMError) as ctx:
                FastEmbedder()
        self.assertIn("模型加载失败", str(ctx.exception))

    def test_embed_returns_plain_floats(self):
        """embed 应返回纯 List[float]（numpy 标量需转 float，便于 JSON 序列化）。"""
        class _Vec:
            def tolist(self):
                return [0.5, -0.25]

        fake_mod = mock.MagicMock()
        fake_mod.TextEmbedding.return_value.embed.return_value = iter([_Vec()])
        with mock.patch.dict("sys.modules", {"fastembed": fake_mod}):
            emb = FastEmbedder()
            vec = emb.embed("任意文本")
        self.assertEqual(vec, [0.5, -0.25])
        self.assertTrue(all(type(x) is float for x in vec))


class TestFastEmbedCacheDir(unittest.TestCase):
    def test_env_override_wins(self):
        with mock.patch.dict(os.environ, {"EMBEDDING_CACHE_DIR": r"D:\short_cache"}):
            self.assertEqual(_default_fastembed_cache(), r"D:\short_cache")

    def test_windows_default_is_short(self):
        env = {k: v for k, v in os.environ.items() if k != "EMBEDDING_CACHE_DIR"}
        with mock.patch.dict(os.environ, env, clear=True):
            path = _default_fastembed_cache()
        if os.name == "nt":
            # Windows 下 HF 缓存含 models--org--model 长目录名，根目录必须短，
            # 否则触发 WinError 206
            self.assertLessEqual(len(path), 20, f"Windows 默认缓存目录过长: {path}")
        else:
            self.assertTrue(path.endswith("fastembed"))


class TestAutoBackendFallback(unittest.TestCase):
    def test_auto_falls_back_to_fastembed(self):
        """sentence_transformers 不可用时应静默回退 fastembed。"""
        sentinel = EmbeddingVectorizer(adapter=mock.MagicMock())

        def _boom(*args, **kwargs):
            raise LLMError("sentence_transformers 未安装")

        with mock.patch("emotion_memory.embeddings.SentenceTransformersEmbedder", side_effect=_boom), \
             mock.patch("emotion_memory.embeddings.FastEmbedder", return_value=sentinel._adapter):
            vec = create_embedding_vectorizer(backend="auto")
        self.assertIsInstance(vec, EmbeddingVectorizer)

    def test_auto_prefers_sentence_transformers(self):
        """sentence_transformers 可用时不应触及 fastembed 分支。"""
        fake_st = mock.MagicMock()
        with mock.patch("emotion_memory.embeddings.SentenceTransformersEmbedder", return_value=fake_st) as m_st, \
             mock.patch("emotion_memory.embeddings.FastEmbedder") as m_fe:
            vec = create_embedding_vectorizer(backend="auto")
        self.assertIs(vec._adapter, fake_st)
        m_st.assert_called_once()
        m_fe.assert_not_called()

    def test_auto_all_fail_raises_aggregate(self):
        def _boom(*args, **kwargs):
            raise LLMError("不可用")

        with mock.patch("emotion_memory.embeddings.SentenceTransformersEmbedder", side_effect=_boom), \
             mock.patch("emotion_memory.embeddings.FastEmbedder", side_effect=_boom):
            with self.assertRaises(LLMError) as ctx:
                create_embedding_vectorizer(backend="auto")
        msg = str(ctx.exception)
        self.assertIn("sentence_transformers", msg)
        self.assertIn("fastembed", msg)

    def test_unknown_backend_lists_new_backends(self):
        with self.assertRaises(LLMError) as ctx:
            create_embedding_vectorizer(backend="no_such_backend")
        self.assertIn("fastembed", str(ctx.exception))
        self.assertIn("auto", str(ctx.exception))


class TestParaphraseQuerySet(unittest.TestCase):
    """改写查询集必须覆盖全部样例且不复用原文字面（否则对照无效）。"""

    def _load(self):
        from benchmark.benchmark import _PARAPHRASES, _SAMPLES
        return _PARAPHRASES, _SAMPLES

    def test_covers_all_samples(self):
        paraphrases, samples = self._load()
        missing = [t for t, _ in samples if t not in paraphrases]
        self.assertEqual(missing, [], f"以下样例缺改写查询: {missing}")

    def test_no_long_lexical_overlap(self):
        paraphrases, _ = self._load()

        def grams(s, n=3):
            return {s[i:i + n] for i in range(len(s) - n + 1)}

        for text, para in paraphrases.items():
            overlap = grams(text) & grams(para)
            self.assertEqual(overlap, set(),
                             f"改写查询与原文存在 {len(overlap)} 个 3 字重合片段: {overlap}")

    def test_paraphrase_mode_used_by_runner(self):
        from benchmark.benchmark import BenchmarkRunner, _PARAPHRASES
        runner = BenchmarkRunner(memories=4, top_k=3, seed=42, query_mode="paraphrase")
        seeds = [("id1", text, 1.0) for text in list(_PARAPHRASES)[:4]]
        queries = runner._make_queries(seeds)
        self.assertEqual(queries, [_PARAPHRASES[t] for _, t, _ in seeds])

    def test_lexical_mode_keeps_original_behavior(self):
        from benchmark.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(memories=4, top_k=3, seed=42, query_mode="lexical")
        seeds = [("id1", "去年生日和三个老同学爬山，很开心", 1.0)]
        q = runner._make_queries(seeds)[0]
        self.assertNotEqual(q, "", "lexical 模式应仍从原文抽字生成查询")


class TestBowDefaultUnaffected(unittest.TestCase):
    def test_default_vectorizer_stays_bow(self):
        from emotion_memory.config import Dial
        from emotion_memory.engine import EmotionalMemorySystem
        import tempfile

        dial = Dial()
        self.assertFalse(dial.embedding_enabled, "embedding 开关默认必须关闭")
        self.assertEqual(dial.embedding_backend, "mock")
        sys_obj = EmotionalMemorySystem(base_dir=tempfile.mkdtemp(prefix="v063_"), seed=1, dial=dial)
        from emotion_memory.store.vector_index import BagOfWordsVectorizer
        self.assertIsInstance(sys_obj._vectorizer, BagOfWordsVectorizer)


if __name__ == "__main__":
    unittest.main()
