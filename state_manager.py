from sqlalchemy import Boolean, create_engine, Column, Integer, String, Text, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import json
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
    confidence = Column(Float, default=1.0)
    is_provisional = Column(Boolean, default=False)

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

        # Migrate plot outline
        if os.path.exists(PLOT_OUTLINE_FILE):
            try:
                with open(PLOT_OUTLINE_FILE, 'r', encoding='utf-8') as f:
                    plot_data = json.load(f)
                    if isinstance(plot_data, dict):
                        # Store plot outline as a WorldBuilding entry with a special name
                        session = self.get_session()
                        plot_entry = WorldBuilding(
                            name="plot_outline",
                            data=json.dumps(plot_data)
                        )
                        
                        # Check if entry already exists
                        existing = session.query(WorldBuilding).filter_by(name="plot_outline").first()
                        if existing:
                            existing.data = json.dumps(plot_data)
                        else:
                            session.add(plot_entry)
                        
                        session.commit()
                        logger.info(f"Migrated plot outline from {PLOT_OUTLINE_FILE} to ORM database")
                    else:
                        logger.warning(f"Plot outline in {PLOT_OUTLINE_FILE} is not a dictionary. Skipping.")
            except Exception as e:
                logger.error(f"Error migrating plot outline: {e}")
        else:
            logger.warning(f"Plot outline file {PLOT_OUTLINE_FILE} not found. Skipping.")

        # Migrate character profiles
        if os.path.exists(CHARACTER_PROFILES_FILE):
            try:
                with open(CHARACTER_PROFILES_FILE, 'r', encoding='utf-8') as f:
                    character_data = json.load(f)
                    if isinstance(character_data, dict):
                        session = self.get_session()
                        
                        # Store each character as a separate entry
                        for char_name, char_info in character_data.items():
                            if not isinstance(char_info, dict):
                                continue
                                
                            char_entry = CharacterProfile(
                                name=char_name,
                                data=json.dumps(char_info)
                            )
                            
                            # Check if entry already exists
                            existing = session.query(CharacterProfile).filter_by(name=char_name).first()
                            if existing:
                                existing.data = json.dumps(char_info)
                            else:
                                session.add(char_entry)
                        
                        session.commit()
                        logger.info(f"Migrated character profiles from {CHARACTER_PROFILES_FILE} to ORM database")
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
                        
                        # Store each world building element as a separate entry
                        for element_name, element_info in world_data.items():
                            if not isinstance(element_info, dict) and not isinstance(element_info, list):
                                continue
                                
                            element_entry = WorldBuilding(
                                name=element_name,
                                data=json.dumps(element_info)
                            )
                            
                            # Check if entry already exists
                            existing = session.query(WorldBuilding).filter_by(name=element_name).first()
                            if existing:
                                existing.data = json.dumps(element_info)
                            else:
                                session.add(element_entry)
                        
                        session.commit()
                        logger.info(f"Migrated world building from {WORLD_BUILDER_FILE} to ORM database")
                    else:
                        logger.warning(f"World building in {WORLD_BUILDER_FILE} is not a dictionary. Skipping.")
            except Exception as e:
                logger.error(f"Error migrating world building: {e}")
        else:
            logger.warning(f"World building file {WORLD_BUILDER_FILE} not found. Skipping.")

        logger.info("Migration from JSON files to ORM database completed")
    
    # Methods to get data from ORM
    def get_plot_outline(self) -> Dict[str, Any]:
        """
        Gets the plot outline from the ORM database.
        Returns a dictionary containing the plot outline data.
        """
        session = self.get_session()
        try:
            plot_entry = session.query(WorldBuilding).filter_by(name="plot_outline").first()
            if plot_entry:
                return json.loads(plot_entry.data)
            else:
                self.logger.warning("Plot outline not found in ORM database. Returning empty dictionary.")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting plot outline from ORM: {e}")
            return {}
        finally:
            session.close()
    
    def get_character_profiles(self) -> Dict[str, Any]:
        """
        Gets all character profiles from the ORM database.
        Returns a dictionary where keys are character names and values are character data.
        """
        session = self.get_session()
        try:
            profiles = {}
            character_entries = session.query(CharacterProfile).all()
            for entry in character_entries:
                profiles[entry.name] = json.loads(entry.data)
            return profiles
        except Exception as e:
            self.logger.error(f"Error getting character profiles from ORM: {e}")
            return {}
        finally:
            session.close()
    
    def get_world_building(self) -> Dict[str, Any]:
        """
        Gets all world building data from the ORM database.
        Returns a dictionary where keys are element names and values are element data.
        Excludes the special "plot_outline" entry.
        """
        session = self.get_session()
        try:
            world_data = {}
            world_entries = session.query(WorldBuilding).filter(WorldBuilding.name != "plot_outline").all()
            for entry in world_entries:
                world_data[entry.name] = json.loads(entry.data)
            return world_data
        except Exception as e:
            self.logger.error(f"Error getting world building from ORM: {e}")
            return {}
        finally:
            session.close()
    
    # Methods to save data to ORM
    def save_plot_outline(self, plot_data: Dict[str, Any]) -> bool:
        """
        Saves the plot outline to the ORM database.
        Returns True if successful, False otherwise.
        """
        if not isinstance(plot_data, dict):
            self.logger.error("Cannot save plot outline: data is not a dictionary")
            return False
        
        session = self.get_session()
        try:
            plot_entry = session.query(WorldBuilding).filter_by(name="plot_outline").first()
            if plot_entry:
                plot_entry.data = json.dumps(plot_data)
            else:
                plot_entry = WorldBuilding(
                    name="plot_outline",
                    data=json.dumps(plot_data)
                )
                session.add(plot_entry)
            
            session.commit()
            self.logger.info("Saved plot outline to ORM database")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving plot outline to ORM: {e}")
            return False
        finally:
            session.close()
    
    def save_character_profile(self, char_name: str, char_data: Dict[str, Any]) -> bool:
        """
        Saves a character profile to the ORM database.
        Returns True if successful, False otherwise.
        """
        if not char_name or not isinstance(char_data, dict):
            self.logger.error("Cannot save character profile: invalid name or data")
            return False
        
        session = self.get_session()
        try:
            char_entry = session.query(CharacterProfile).filter_by(name=char_name).first()
            if char_entry:
                char_entry.data = json.dumps(char_data)
            else:
                char_entry = CharacterProfile(
                    name=char_name,
                    data=json.dumps(char_data)
                )
                session.add(char_entry)
            
            session.commit()
            self.logger.info(f"Saved character profile for '{char_name}' to ORM database")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving character profile to ORM: {e}")
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
    
    def save_world_building_element(self, element_name: str, element_data: Any) -> bool:
        """
        Saves a world building element to the ORM database.
        Returns True if successful, False otherwise.
        """
        if not element_name or not (isinstance(element_data, dict) or isinstance(element_data, list)):
            self.logger.error("Cannot save world building element: invalid name or data")
            return False
        
        session = self.get_session()
        try:
            element_entry = session.query(WorldBuilding).filter_by(name=element_name).first()
            if element_entry:
                element_entry.data = json.dumps(element_data)
            else:
                element_entry = WorldBuilding(
                    name=element_name,
                    data=json.dumps(element_data)
                )
                session.add(element_entry)
            
            session.commit()
            self.logger.info(f"Saved world building element '{element_name}' to ORM database")
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving world building element to ORM: {e}")
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
