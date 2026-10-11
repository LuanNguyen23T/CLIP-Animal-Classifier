import os
import sys
import re

# Add the backend directory to sys.path so we can import from app
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine, Base
from app.models.species import Species
from app.models.breed import Breed

def seed_data():
    db = SessionLocal()
    
    # 1. Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # 2. Seed Species
    species_cat = db.query(Species).filter(Species.name == "Cat").first()
    if not species_cat:
        species_cat = Species(id=1, name="Cat")
        db.add(species_cat)
        
    species_dog = db.query(Species).filter(Species.name == "Dog").first()
    if not species_dog:
        species_dog = Species(id=2, name="Dog")
        db.add(species_dog)
        
    db.commit()
    print("Seeded species: Cat, Dog")
    
    # 3. Seed Breeds from list.txt
    dataset_list_path = os.path.join(os.path.dirname(__file__), '../dataset/oxford-iiit-pet/annotations/list.txt')
    
    if not os.path.exists(dataset_list_path):
        print(f"File not found: {dataset_list_path}")
        return
        
    with open(dataset_list_path, 'r') as f:
        lines = f.readlines()
        
    breed_names = set()
    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) >= 4:
            # Image CLASS-ID SPECIES BREED ID
            # Example: Abyssinian_100 1 1 1
            # The class name is the image name minus the last underscore and number
            image_name = parts[0]
            class_id = int(parts[1])
            species_id = int(parts[2]) # 1:Cat 2:Dog
            
            # Extract breed name (remove trailing _number)
            match = re.match(r'^(.*)_\d+$', image_name)
            if match:
                breed_name = match.group(1)
                
                # Check if we already processed this class_id
                if class_id not in [b.id for b in db.query(Breed).all()]:
                    existing_breed = db.query(Breed).filter(Breed.id == class_id).first()
                    if not existing_breed:
                        new_breed = Breed(id=class_id, name=breed_name, species_id=species_id)
                        db.add(new_breed)
                        db.commit()
                        print(f"Added breed: {breed_name} (ID: {class_id}, Species: {species_id})")
                        
    print("Database seeding completed.")
    db.close()

if __name__ == "__main__":
    seed_data()
