from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid

# Configuração do Banco de Dados MySQL
USERNAME = "seu_usuario"  # Substitua pelo seu usuário do MySQL
PASSWORD = "sua_senha"  # Substitua pela sua senha
HOST = "localhost"  # Ou o IP do seu servidor MySQL
DATABASE = "mental_health_db"

DATABASE_URL = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"

engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# Criar tabelas no MySQL
Base.metadata.create_all(engine)

# Criar uma sessão para interagir com o banco de dados
def get_db_session():
    return SessionLocal()
