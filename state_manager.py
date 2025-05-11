from sqlalchemy import Boolean, create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import json
from typing import Optional, List, Dict
from config import DATABASE_FILE

Base = declarative_base()

class Chapter(Base):
    __tablename__ = 'chapters'
    
    id = Column(Integer, primary_key=True)
    number = Column(Integer, unique=True, nullable=False)
    text = Column(Text, nullable=False)
    raw_llm_log = Column(Text)
    summary = Column(Text)
    is_provisional = Column(Boolean, default=False)

class CharacterProfile(Base):
    __tablename__ = 'character_profiles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    data = Column(Text, nullable=False)  # Stored as JSON string

class WorldBuilding(Base):
    __tablename__ = 'world_building'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    data = Column(Text, nullable=False)  # Stored as JSON string

class KnowledgeGraphTriple(Base):
    __tablename__ = 'knowledge_graph'
    
    id = Column(Integer, primary_key=True)
    subject = Column(String(255))
    predicate = Column(String(255))
    obj = Column(String(255))
    chapter_number = Column(Integer)
    confidence = Column(float, default=1.0)
    is_provisional = Column(Boolean, default=False)

class StateManager:
    def __init__(self):
        self.engine = create_engine(f'sqlite:///{DATABASE_FILE}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.Session()
    
    def migrate_from_json(self):
        # Implementation for JSON to ORM migration
        pass

# Initialize singleton instance
state_manager = StateManager()
