from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# La URL de TiDB se obtendrá de variables de entorno en producción (Railway)
# En local, se usarán por defecto tus credenciales de TiDB Cloud
TIDB_URL = "mysql+pymysql://2czmxYCNTp9AbwJ.root:mR8V2hc3yVyRKyny@gateway01.us-east-1.prod.aws.tidbcloud.com:4000/cul_chatbot"
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", TIDB_URL)

# Para TiDB Cloud Serverless necesitamos conectarnos por SSL
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"ssl": {"ca": "/etc/ssl/certs/ca-certificates.crt"}}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
