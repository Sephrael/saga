# state_manager.py
from sqlalchemy import Boolean, Column, Integer, String, Text, ForeignKey, Float, Index, LargeBinary
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.future import select
import logging
import json
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

import config # For DATABASE_FILE, EMBEDDING_DTYPE, etc.

Base = declarative_base()
logger = logging.getLogger(__name__)

# --- ORM MODELS ---

class Chapter(Base):
    __tablename__ = 'chapters'
    
    chapter_number = Column(Integer, primary_key=True, autoincrement=False) # Explicitly not auto-incrementing
    text = Column(Text)
    raw_text = Column(Text) 
    summary = Column(Text)
    is_provisional = Column(Boolean, default=False)
    
    embedding = relationship("Embedding", back_populates="chapter", uselist=False, cascade="all, delete-orphan")
    # Relationship for knowledge_triples if chapter_added is a FK
    # knowledge_triples = relationship("KnowledgeGraphTriple", back_populates="chapter_ref")


class Embedding(Base):
    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    chapter_number = Column(Integer, ForeignKey('chapters.chapter_number', ondelete="CASCADE"), unique=True, nullable=False)
    embedding_blob = Column(LargeBinary, nullable=False)
    dtype = Column(String, nullable=False)
    shape = Column(String, nullable=False) 

    chapter = relationship("Chapter", back_populates="embedding")

class KnowledgeGraphTriple(Base):
    __tablename__ = 'knowledge_graph' # This matches the old table name
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    subject = Column(String(255), nullable=False)
    predicate = Column(String(255), nullable=False)
    obj = Column(String(255), nullable=False) # Corrected from 'object'
    chapter_added = Column(Integer, nullable=False) 
    confidence = Column(Float, default=1.0)
    is_provisional = Column(Boolean, default=False)

    # If chapter_added is a foreign key to chapters.chapter_number:
    # chapter_ref_id = Column(Integer, ForeignKey('chapters.chapter_number'))
    # chapter_ref = relationship("Chapter", back_populates="knowledge_triples")

    __table_args__ = (
        Index('idx_kg_subject', 'subject'),
        Index('idx_kg_predicate', 'predicate'),
        Index('idx_kg_obj', 'obj'), # Index on the correct column name
        Index('idx_kg_spo', 'subject', 'predicate', 'obj'),
        Index('idx_kg_chapter_added', 'chapter_added'),
    )

class OrmPlotOutline(Base):
    __tablename__ = 'orm_plot_outline'
    id = Column(Integer, primary_key=True)
    key = Column(String(255), unique=True, nullable=False, default="main_plot")
    value = Column(Text, nullable=False) 

class OrmCharacterProfile(Base):
    __tablename__ = 'orm_character_profiles'
    id = Column(Integer, primary_key=True)
    # Store all profiles as a single JSON blob for simplicity, matching current agent structure
    key = Column(String(255), unique=True, nullable=False, default="_all_profiles_")
    data = Column(Text, nullable=False) 

class OrmWorldBuilding(Base):
    __tablename__ = 'orm_world_building'
    id = Column(Integer, primary_key=True)
    # Store all world_building data as a single JSON blob
    key = Column(String(255), unique=True, nullable=False, default="main_world")
    data = Column(Text, nullable=False)


class state_managerSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(state_managerSingleton, cls).__new__(cls)
            cls._instance._initialized_flag = False # Use a different name to avoid conflicts
        return cls._instance

    def __init__(self):
        if self._initialized_flag:
            return
        
        self.logger = logging.getLogger(__name__)
        self.engine = create_async_engine(f'sqlite+aiosqlite:///{config.DATABASE_FILE}')
        
        self.AsyncSessionLocal = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False, # Recommended for async
            autocommit=False # Recommended for async
        )
        self._initialized_flag = True
        self.logger.info(f"Async state_managerSingleton initialized with engine for {config.DATABASE_FILE}")

    async def create_db_and_tables(self):
        async with self.engine.begin() as conn:
            # To ensure a clean state during transition, you might want to drop tables.
            # WARNING: This will delete all existing data. Use with caution.
            # await conn.run_sync(Base.metadata.drop_all)
            # logger.info("Dropped all tables for a fresh start.")
            await conn.run_sync(Base.metadata.create_all)
        self.logger.info("Database tables created/verified via ORM based on current models.")

    def _serialize_embedding(self, embedding: np.ndarray) -> Tuple[bytes, str, str]:
        embedding_to_save = embedding.astype(config.EMBEDDING_DTYPE)
        if embedding_to_save.ndim == 0:
             embedding_to_save = embedding_to_save.reshape(1)
        return embedding_to_save.tobytes(), str(embedding_to_save.dtype), json.dumps(list(embedding_to_save.shape))

    def _deserialize_embedding(self, blob: bytes, dtype_str: str, shape_str: str) -> Optional[np.ndarray]:
        try:
            shape = tuple(json.loads(shape_str))
            dtype = np.dtype(dtype_str)
            return np.frombuffer(blob, dtype=dtype).reshape(shape)
        except Exception as e:
            self.logger.error(f"Error deserializing embedding (shape: {shape_str}, dtype: {dtype_str}): {e}", exc_info=True)
            return None

    async def save_plot_outline(self, plot_data: Dict[str, Any]) -> bool:
        async with self.AsyncSessionLocal() as session:
            async with session.begin():
                stmt = select(OrmPlotOutline).where(OrmPlotOutline.key == "main_plot")
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()
                json_value = json.dumps(plot_data, ensure_ascii=False)
                if entry:
                    entry.value = json_value
                else:
                    entry = OrmPlotOutline(key="main_plot", value=json_value)
                    session.add(entry)
            # session.begin() handles commit/rollback automatically
        self.logger.info("Saved plot outline to ORM.")
        return True

    async def get_plot_outline(self) -> Dict[str, Any]:
        async with self.AsyncSessionLocal() as session:
            stmt = select(OrmPlotOutline).where(OrmPlotOutline.key == "main_plot")
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()
            if entry and entry.value:
                return json.loads(entry.value)
            return {}

    async def save_character_profiles(self, profiles_data: Dict[str, Any]) -> bool:
        async with self.AsyncSessionLocal() as session:
            async with session.begin():
                stmt = select(OrmCharacterProfile).where(OrmCharacterProfile.key == "_all_profiles_")
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()
                json_data = json.dumps(profiles_data, ensure_ascii=False)
                if entry:
                    entry.data = json_data
                else:
                    entry = OrmCharacterProfile(key="_all_profiles_", data=json_data)
                    session.add(entry)
        self.logger.info("Saved character profiles to ORM.")
        return True

    async def get_character_profiles(self) -> Dict[str, Any]:
        async with self.AsyncSessionLocal() as session:
            stmt = select(OrmCharacterProfile).where(OrmCharacterProfile.key == "_all_profiles_")
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()
            if entry and entry.data:
                return json.loads(entry.data)
            return {}

    async def save_world_building(self, world_data: Dict[str, Any]) -> bool:
        async with self.AsyncSessionLocal() as session:
            async with session.begin():
                stmt = select(OrmWorldBuilding).where(OrmWorldBuilding.key == "main_world")
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()
                json_data = json.dumps(world_data, ensure_ascii=False)
                if entry:
                    entry.data = json_data
                else:
                    entry = OrmWorldBuilding(key="main_world", data=json_data)
                    session.add(entry)
        self.logger.info("Saved world building to ORM.")
        return True

    async def get_world_building(self) -> Dict[str, Any]:
        async with self.AsyncSessionLocal() as session:
            stmt = select(OrmWorldBuilding).where(OrmWorldBuilding.key == "main_world")
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()
            if entry and entry.data:
                return json.loads(entry.data)
            return {}

    async def async_load_chapter_count(self) -> int:
        async with self.AsyncSessionLocal() as session:
            stmt = select(Chapter.chapter_number).where(Chapter.chapter_number > 0).order_by(Chapter.chapter_number.desc()).limit(1)
            result = await session.execute(stmt)
            max_chap_num = result.scalar_one_or_none()
            count = max_chap_num if max_chap_num is not None else 0
            self.logger.info(f"Async ORM loaded chapter count: {count}")
            return count


    async def async_save_chapter_data(self, chapter_number: int, text: str, raw_llm_output: str, summary: Optional[str], embedding_array: Optional[np.ndarray], is_provisional: bool = False):
        if chapter_number <= 0:
            self.logger.error(f"Async ORM: Cannot save chapter data for invalid chapter_number: {chapter_number}.")
            return

        async with self.AsyncSessionLocal() as session:
            async with session.begin():
                stmt = select(Chapter).where(Chapter.chapter_number == chapter_number).options(relationship(Chapter.embedding))
                result = await session.execute(stmt)
                chapter_obj = result.scalar_one_or_none()

                if not chapter_obj:
                    chapter_obj = Chapter(chapter_number=chapter_number)
                    session.add(chapter_obj)
                
                chapter_obj.text = text
                chapter_obj.raw_text = raw_llm_output
                chapter_obj.summary = summary if summary is not None else ""
                chapter_obj.is_provisional = is_provisional

                if embedding_array is not None and isinstance(embedding_array, np.ndarray) and embedding_array.size > 0:
                    blob, dtype_str, shape_str = self._serialize_embedding(embedding_array)
                    
                    if chapter_obj.embedding: # Existing embedding object
                        embedding_obj = chapter_obj.embedding
                    else: # New embedding object
                        embedding_obj = Embedding(chapter_number=chapter_number) # chapter_number FK is crucial
                        chapter_obj.embedding = embedding_obj # Link to chapter
                        session.add(embedding_obj)


                    embedding_obj.embedding_blob = blob
                    embedding_obj.dtype = dtype_str
                    embedding_obj.shape = shape_str
                elif chapter_obj.embedding: # No new embedding, but an old one exists
                    await session.delete(chapter_obj.embedding)
                    chapter_obj.embedding = None
        self.logger.info(f"Async ORM: Successfully saved chapter data for chapter {chapter_number}.")


    async def async_get_chapter_data_from_db(self, chapter_number: int) -> Optional[Dict[str, Any]]:
        if chapter_number <= 0: return None
        async with self.AsyncSessionLocal() as session:
            stmt = select(Chapter).where(Chapter.chapter_number == chapter_number)
            result = await session.execute(stmt)
            chapter_obj = result.scalar_one_or_none()
            if chapter_obj:
                return {
                    "text": chapter_obj.text,
                    "summary": chapter_obj.summary,
                    "is_provisional": chapter_obj.is_provisional,
                    "raw_text": chapter_obj.raw_text
                }
            self.logger.debug(f"Async ORM: No data found for chapter {chapter_number}.")
            return None

    async def async_get_embedding_from_db(self, chapter_number: int) -> Optional[np.ndarray]:
        if chapter_number <= 0: return None
        async with self.AsyncSessionLocal() as session:
            stmt = select(Embedding).where(Embedding.chapter_number == chapter_number)
            result = await session.execute(stmt)
            embedding_obj = result.scalar_one_or_none()
            if embedding_obj:
                return self._deserialize_embedding(embedding_obj.embedding_blob, embedding_obj.dtype, embedding_obj.shape)
            self.logger.debug(f"Async ORM: No embedding found for chapter {chapter_number}.")
            return None

    async def async_get_all_past_embeddings(self, current_chapter_number: int) -> List[Tuple[int, np.ndarray]]:
        embeddings_list: List[Tuple[int, np.ndarray]] = []
        async with self.AsyncSessionLocal() as session:
            stmt = (
                select(Embedding)
                .where(Embedding.chapter_number < current_chapter_number, Embedding.chapter_number > 0)
                .order_by(Embedding.chapter_number.desc())
            )
            results = await session.execute(stmt)
            for row_embedding_obj in results.scalars().all():
                deserialized_emb = self._deserialize_embedding(row_embedding_obj.embedding_blob, row_embedding_obj.dtype, row_embedding_obj.shape)
                if deserialized_emb is not None:
                    embeddings_list.append((row_embedding_obj.chapter_number, deserialized_emb))
        self.logger.info(f"Async ORM: Retrieved {len(embeddings_list)} past embeddings.")
        return embeddings_list
        
    async def async_add_kg_triple(self, subject: str, predicate: str, obj_val: str, chapter_added: int, confidence: float = 1.0, is_provisional: bool = False):
        subj_s, pred_s, obj_s = subject.strip(), predicate.strip(), obj_val.strip()
        if not all([subj_s, pred_s, obj_s]) or chapter_added < config.KG_PREPOPULATION_CHAPTER_NUM:
            self.logger.warning(f"Async ORM: Invalid KG triple for add: S='{subj_s}', P='{pred_s}', O='{obj_s}', Chap={chapter_added}")
            return

        async with self.AsyncSessionLocal() as session:
            async with session.begin():
                stmt_exists = select(KnowledgeGraphTriple).where(
                    KnowledgeGraphTriple.subject == subj_s,
                    KnowledgeGraphTriple.predicate == pred_s,
                    KnowledgeGraphTriple.obj == obj_s,
                    KnowledgeGraphTriple.chapter_added == chapter_added
                )
                result_exists = await session.execute(stmt_exists)
                if result_exists.scalar_one_or_none():
                    self.logger.debug(f"Async ORM: Triple ({subj_s}, {pred_s}, {obj_s}) for Ch {chapter_added} already exists. Skipping.")
                    return

                new_triple = KnowledgeGraphTriple(
                    subject=subj_s, predicate=pred_s, obj=obj_s,
                    chapter_added=chapter_added, confidence=confidence, is_provisional=is_provisional
                )
                session.add(new_triple)
        self.logger.debug(f"Async ORM: Added KG triple for Ch {chapter_added}: ({subj_s}, {pred_s}, {obj_s}).")


    async def async_query_kg(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj_val: Optional[str] = None, chapter_limit: Optional[int] = None, include_provisional: bool = True) -> List[Dict[str, Any]]:
        async with self.AsyncSessionLocal() as session:
            stmt = select(KnowledgeGraphTriple)
            if subject is not None:
                stmt = stmt.where(KnowledgeGraphTriple.subject == subject.strip())
            if predicate is not None:
                stmt = stmt.where(KnowledgeGraphTriple.predicate == predicate.strip())
            if obj_val is not None:
                stmt = stmt.where(KnowledgeGraphTriple.obj == obj_val.strip())
            if chapter_limit is not None: # Allow 0 for KG_PREPOPULATION_CHAPTER_NUM
                stmt = stmt.where(KnowledgeGraphTriple.chapter_added <= chapter_limit)
            if not include_provisional:
                stmt = stmt.where(KnowledgeGraphTriple.is_provisional == False) # SQLAlchemy uses Python bools
            
            stmt = stmt.order_by(KnowledgeGraphTriple.chapter_added.desc(), KnowledgeGraphTriple.confidence.desc())
            
            results = await session.execute(stmt)
            triples_list: List[Dict[str, Any]] = []
            for triple_obj in results.scalars().all():
                triples_list.append({
                    "id": triple_obj.id,
                    "subject": triple_obj.subject,
                    "predicate": triple_obj.predicate,
                    "object": triple_obj.obj, # Use 'object' key for consistency with what DatabaseManager returned
                    "chapter_added": triple_obj.chapter_added,
                    "confidence": triple_obj.confidence,
                    "is_provisional": triple_obj.is_provisional
                })
            self.logger.debug(f"Async ORM: KG query returned {len(triples_list)} results.")
            return triples_list

    async def async_get_most_recent_value(self, subject: str, predicate: str, chapter_limit: Optional[int] = None, include_provisional: bool = False) -> Optional[str]:
        if not subject.strip() or not predicate.strip():
            self.logger.warning(f"Async ORM: get_most_recent_value: empty subject or predicate. S='{subject}', P='{predicate}'")
            return None
        
        results = await self.async_query_kg(subject=subject, predicate=predicate, chapter_limit=chapter_limit, include_provisional=include_provisional)
        if results:
            value = str(results[0]["object"]) 
            self.logger.debug(f"Async ORM: Found most recent value: '{value}' from Ch {results[0]['chapter_added']}")
            return value
        self.logger.debug(f"Async ORM: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}")
        return None

state_manager = state_managerSingleton()