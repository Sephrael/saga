from sqlalchemy import Boolean, create_engine, Column, Integer, String, Text, ForeignKey, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import logging
from typing import Optional, List, Dict, Any
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
    
    # Relationships
    knowledge_triples = relationship("KnowledgeGraphTriple", back_populates="chapter")
    character_relations = relationship("CharacterRelation", back_populates="chapter")

class CharacterProfile(Base):
    __tablename__ = 'character_profiles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    
    # Relationships
    attributes = relationship("CharacterAttribute", back_populates="profile")
    relations = relationship("CharacterRelation", back_populates="profile")

class CharacterAttribute(Base):
    __tablename__ = 'character_attributes'
    
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('character_profiles.id'), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)
    
    # Relationships
    profile = relationship("CharacterProfile", back_populates="attributes")
    
    __table_args__ = (Index('ix_profile_key', 'profile_id', 'key'),)

class WorldBuilding(Base):
    __tablename__ = 'world_building'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    
    # Relationships
    elements = relationship("WorldElement", back_populates="building")

class WorldElement(Base):
    __tablename__ = 'world_elements'
    
    id = Column(Integer, primary_key=True)
    building_id = Column(Integer, ForeignKey('world_building.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    
    # Relationships
    building = relationship("WorldBuilding", back_populates="elements")
    properties = relationship("WorldProperty", back_populates="element")
    
    __table_args__ = (Index('ix_building_name', 'building_id', 'name'),)

class WorldProperty(Base):
    __tablename__ = 'world_properties'
    
    id = Column(Integer, primary_key=True)
    element_id = Column(Integer, ForeignKey('world_elements.id'), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)
    
    # Relationships
    element = relationship("WorldElement", back_populates="properties")
    
    __table_args__ = (Index('ix_element_key', 'element_id', 'key'),)

class KnowledgeGraphTriple(Base):
    __tablename__ = 'knowledge_graph'
    
    id = Column(Integer, primary_key=True)
    subject = Column(String(255))
    predicate = Column(String(255))
    obj = Column(String(255))
    chapter_number = Column(Integer, ForeignKey('chapters.number'))
    confidence = Column(Float, default=1.0)
    is_provisional = Column(Boolean, default=False)
    
    # Relationships
    chapter = relationship("Chapter", back_populates="knowledge_triples")

class CharacterRelation(Base):
    __tablename__ = 'character_relations'
    
    id = Column(Integer, primary_key=True)
    chapter_number = Column(Integer, ForeignKey('chapters.number'), nullable=False)
    from_character_id = Column(Integer, ForeignKey('character_profiles.id'), nullable=False)
    to_character_id = Column(Integer, ForeignKey('character_profiles.id'), nullable=False)
    relation_type = Column(String(255), nullable=False)
    strength = Column(Float, default=1.0)
    
    # Relationships
    chapter = relationship("Chapter", back_populates="character_relations")
    from_character = relationship("CharacterProfile", foreign_keys=[from_character_id])
    to_character = relationship("CharacterProfile", foreign_keys=[to_character_id])
class StateManager:
    def __init__(self):
        self.engine = create_engine(f'sqlite:///{DATABASE_FILE}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)
    
    def get_session(self):
        return self.Session()
    
    def migrate_from_json(self):
        """
        Migrates data from JSON files to the ORM database.
        Reads plot_outline.json, character_profiles.json, and world_building.json,
        and converts the data to ORM models.
        """
        import os
        import json
        import logging
        from config import PLOT_OUTLINE_FILE, CHARACTER_PROFILES_FILE, WORLD_BUILDER_FILE

        logger = logging.getLogger(__name__)
        logger.info("Starting migration from JSON files to ORM database...")

        # Migrate plot outline using relational models
        if os.path.exists(PLOT_OUTLINE_FILE):
            try:
                with open(PLOT_OUTLINE_FILE, 'r', encoding='utf-8') as f:
                    plot_data = json.load(f)
                    if isinstance(plot_data, dict):
                        session = self.get_session()
                        
                        # Use the same pattern as world building elements
                        wb_entry = WorldBuilding(name="plot_outline")
                        session.add(wb_entry)
                        session.commit()  # Need ID for relationships
                        
                        we_entry = WorldElement(
                            building_id=wb_entry.id,
                            name="plot_outline",
                            description="Plot outline data"
                        )
                        session.add(we_entry)
                        
                        # Add properties from plot_data
                        for key, value in plot_data.items():
                            session.add(WorldProperty(
                                element_id=we_entry.id,
                                key=key,
                                value=str(value)
                            ))
                        
                        session.commit()
                        logger.info(f"Migrated plot outline from {PLOT_OUTLINE_FILE} using relational models")
                    else:
                        logger.warning(f"Plot outline in {PLOT_OUTLINE_FILE} is not a dictionary. Skipping.")
            except Exception as e:
                logger.error(f"Error migrating plot outline: {e}")
        else:
            logger.warning(f"Plot outline file {PLOT_OUTLINE_FILE} not found. Skipping.")

        # Migrate character profiles using relational models
        if os.path.exists(CHARACTER_PROFILES_FILE):
            try:
                with open(CHARACTER_PROFILES_FILE, 'r', encoding='utf-8') as f:
                    character_data = json.load(f)
                    if isinstance(character_data, dict):
                        session = self.get_session()
                        
                        # Store each character using relational model
                        for char_name, char_info in character_data.items():
                            if not isinstance(char_info, dict):
                                continue
                                
                            # Get or create character profile
                            char_entry = session.query(CharacterProfile).filter_by(name=char_name).first()
                            if not char_entry:
                                char_entry = CharacterProfile(name=char_name)
                                session.add(char_entry)
                                session.commit()  # Need ID for attributes
							
                            # Add attributes from character_info
                            for key, value in char_info.items():
                                attr = session.query(CharacterAttribute).filter_by(
                                    profile_id=char_entry.id,
                                    key=key
                                ).first()
								
                                if attr:
                                    attr.value = str(value)
                                else:
                                    session.add(CharacterAttribute(
                                        profile_id=char_entry.id,
                                        key=key,
                                        value=str(value)
                                    ))
							
                            # Commit after each character to avoid large transaction
                            session.commit()
                        
                        logger.info(f"Migrated character profiles from {CHARACTER_PROFILES_FILE} using relational models")
                    else:
                        logger.warning(f"Character profiles in {CHARACTER_PROFILES_FILE} is not a dictionary. Skipping.")
            except Exception as e:
                logger.error(f"Error migrating character profiles: {e}")
        else:
            logger.warning(f"Character profiles file {CHARACTER_PROFILES_FILE} not found. Skipping.")

        # Migrate world building
        if os.path.exists(WORLD_BUILDER_FILE):
            try:
                with open(WORLD_BUILDER_FILE, 'r', encoding='utf-8') as f:
                    world_data = json.load(f)
                    if isinstance(world_data, dict):
                        session = self.get_session()
                        
                        # Store each world building element using relational models
                        for element_name, element_info in world_data.items():
                            if not isinstance(element_info, dict):
                                continue
                                
                            self.save_world_building_element(element_name, element_info)
                        
                        session.commit()
                        logger.info(f"Migrated world building from {WORLD_BUILDER_FILE} to ORM database")
                    else:
                        logger.warning(f"World building in {WORLD_BUILDER_FILE} is not a dictionary. Skipping.")
            except Exception as e:
                logger.error(f"Error migrating world building: {e}")
                session.rollback()
            finally:
                if 'session' in locals():
                    session.close()
        else:
            logger.warning(f"World building file {WORLD_BUILDER_FILE} not found. Skipping.")

        logger.info("Migration from JSON files to ORM database completed")
    
    # Methods to get data from ORM
    def get_plot_outline(self) -> Dict[str, Any]:
        """
        Gets the plot outline from the ORM database using relational models.
        Returns a dictionary containing the plot outline data.
        """
        session = self.get_session()
        try:
            # Query the WorldBuilding entry
            wb_entry = session.query(WorldBuilding).filter_by(name="plot_outline").first()
            if not wb_entry:
                self.logger.warning("Plot outline not found in ORM database. Returning empty dictionary.")
                return {}
            
            # Get associated WorldElement (should be exactly one)
            we_entry = session.query(WorldElement).filter_by(
                building_id=wb_entry.id,
                name="plot_outline"
            ).first()
            
            if not we_entry:
                self.logger.warning("WorldElement for plot outline not found. Returning empty dictionary.")
                return {}
            
            # Get all properties
            properties = session.query(WorldProperty).filter_by(element_id=we_entry.id).all()
            return {prop.key: prop.value for prop in properties}
            
        except Exception as e:
            self.logger.error(f"Error getting plot outline from ORM: {e}")
            return {}
        finally:
            session.close()
    
    def get_character_profiles(self) -> Dict[str, Any]:
        """
        Gets all character profiles from the ORM database.
        Returns a dictionary where keys are character names and values are full CharacterProfile objects.
        """
        session = self.get_session()
        try:
            return {cp.name: cp for cp in session.query(CharacterProfile).all()}
        except Exception as e:
            self.logger.error(f"Error getting character profiles from ORM: {e}")
            return {}
        finally:
            session.close()
    
    def get_world_building(self) -> Dict[str, Any]:
        """
        Gets all world building data from the ORM database using relational models.
        Returns a dictionary where keys are element names and values are element data.
        Excludes the special "plot_outline" entry.
        """
        session = self.get_session()
        try:
            world_data = {}
            # Get all WorldBuilding entries except plot_outline
            wb_entries = session.query(WorldBuilding).filter(WorldBuilding.name != "plot_outline").all()
            
            for wb_entry in wb_entries:
                # Find associated WorldElement (should be exactly one per WorldBuilding)
                we_entry = session.query(WorldElement).filter_by(
                    building_id=wb_entry.id,
                    name=wb_entry.name
                ).first()
                
                if not we_entry:
                    self.logger.warning(f"WorldElement for {wb_entry.name} not found. Skipping.")
                    continue
                
                # Get all properties for this element
                properties = session.query(WorldProperty).filter_by(element_id=we_entry.id).all()
                world_data[wb_entry.name] = {prop.key: prop.value for prop in properties}
            
            return world_data
        except Exception as e:
            self.logger.error(f"Error getting world building from ORM: {e}")
            return {}
        finally:
            session.close()
    
    # Methods to save data to ORM
    def save_plot_outline(self, plot_data: Dict[str, Any]) -> bool:
        """
        Saves the plot outline to the ORM database using relational models.
        Returns True if successful, False otherwise.
        """
        if not isinstance(plot_data, dict):
            self.logger.error("Cannot save plot outline: data is not a dictionary")
            return False
        
        session = self.get_session()
        try:
            # Get or create WorldBuilding entry
            wb_entry = session.query(WorldBuilding).filter_by(name="plot_outline").first()
            if not wb_entry:
                wb_entry = WorldBuilding(name="plot_outline")
                session.add(wb_entry)
                session.commit()  # Need ID for relationships
            
            # Get or create WorldElement
            we_entry = session.query(WorldElement).filter_by(
                building_id=wb_entry.id,
                name="plot_outline"
            ).first()
            
            if not we_entry:
                we_entry = WorldElement(
                    building_id=wb_entry.id,
                    name="plot_outline",
                    description="Plot outline data"
                )
                session.add(we_entry)
                session.commit()  # Need ID for properties
            
            # Update properties
            for key, value in plot_data.items():
                prop = session.query(WorldProperty).filter_by(
                    element_id=we_entry.id,
                    key=key
                ).first()
                
                if prop:
                    prop.value = str(value)
                else:
                    session.add(WorldProperty(
                        element_id=we_entry.id,
                        key=key,
                        value=str(value)
                    ))
            
            session.commit()
            self.logger.info("Saved plot outline using relational models")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving plot outline: {e}")
            return False
        finally:
            session.close()
    
    def save_character_profile(self, char_name: str, char_data: Dict[str, Any]) -> bool:
        """
        Saves a character profile using relational models.
        Returns True if successful, False otherwise.
        """
        if not char_name or not isinstance(char_data, dict):
            self.logger.error("Cannot save character profile: invalid name or data")
            return False
        
        session = self.get_session()
        try:
            # Get or create character profile
            char_entry = session.query(CharacterProfile).filter_by(name=char_name).first()
            if not char_entry:
                char_entry = CharacterProfile(name=char_name)
                session.add(char_entry)
            
            # Update attributes
            for key, value in char_data.items():
                attr = session.query(CharacterAttribute).filter_by(
                    profile_id=char_entry.id,
                    key=key
                ).first()
                
                if attr:
                    attr.value = str(value)
                else:
                    session.add(CharacterAttribute(
                        profile_id=char_entry.id,
                        key=key,
                        value=str(value)
                    ))
            
            session.commit()
            self.logger.info(f"Saved character profile for '{char_name}' using relational models")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving character profile: {e}")
            return False
        finally:
            session.close()
    
    def save_character_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> bool:
        """
        Saves multiple character profiles to the ORM database.
        Returns True if all saves were successful, False otherwise.
        """
        if not isinstance(profiles, dict):
            self.logger.error("Cannot save character profiles: data is not a dictionary")
            return False
        
        success = True
        for char_name, char_data in profiles.items():
            if not self.save_character_profile(char_name, char_data):
                success = False
        
        return success
    
    def save_world_building_element(self, element_name: str, element_data: Dict[str, Any]) -> bool:
        """
        Saves a world building element using relational models.
        Returns True if successful, False otherwise.
        """
        if not element_name or not isinstance(element_data, dict):
            self.logger.error("Cannot save world building element: invalid name or data")
            return False
        
        session = self.get_session()
        try:
            # Get or create WorldBuilding entry
            wb_entry = session.query(WorldBuilding).filter_by(name=element_name).first()
            if not wb_entry:
                wb_entry = WorldBuilding(name=element_name)
                session.add(wb_entry)
            
            # Get or create WorldElement
            we_entry = session.query(WorldElement).filter_by(
                building_id=wb_entry.id,
                name=element_name
            ).first()
            
            if not we_entry:
                we_entry = WorldElement(
                    building_id=wb_entry.id,
                    name=element_name,
                    description=element_data.get('description', '')
                )
                session.add(we_entry)
            
            # Update properties
            for key, value in element_data.items():
                if key == 'description':
                    continue  # Handled separately
                
                prop = session.query(WorldProperty).filter_by(
                    element_id=we_entry.id,
                    key=key
                ).first()
                
                if prop:
                    prop.value = str(value)
                else:
                    session.add(WorldProperty(
                        element_id=we_entry.id,
                        key=key,
                        value=str(value)
                    ))
            
            session.commit()
            self.logger.info(f"Saved world building element '{element_name}' using relational models")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving world building element: {e}")
            return False
        finally:
            session.close()
    
    def save_world_building(self, world_data: Dict[str, Any]) -> bool:
        """
        Saves multiple world building elements to the ORM database.
        Returns True if all saves were successful, False otherwise.
        """
        if not isinstance(world_data, dict):
            self.logger.error("Cannot save world building: data is not a dictionary")
            return False
        
        success = True
        for element_name, element_data in world_data.items():
            if not self.save_world_building_element(element_name, element_data):
                success = False
        
        return success

# Initialize singleton instance
state_manager = StateManager()
