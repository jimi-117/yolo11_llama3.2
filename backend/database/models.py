from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, func
from sqlalchemy.orm import relationship
from ...frontend.auth.session import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    feedbacks = relationship("Feedback", back_populates="user")

class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now())
    image_name = Column(String)
    inference_time = Column(Float)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="feedbacks")