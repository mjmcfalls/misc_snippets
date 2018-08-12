import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sqlDb = r'sqlite:///testDb.sqlite3'
Base = declarative_base()

class customer(Base):
    __tablename__ = "customerTbl"
    customerTblId = Column("customerTblId", Integer, primary_key=True)
    customerName = Column("customerName", String)
    addressId = Column(Integer, ForeignKey("addressTbl.addressTblId"))
    addresses = relationship("address", back_populates="customers")

class address(Base):
    __tablename__ = "addressTbl"
    addressTblId = Column("addressTblId", Integer, primary_key=True)
    addressName = Column("address", String)
    cityId = Column("cityId", ForeignKey("cityTbl.cityTblId"))
    customers = relationship("customer")
    cities = relationship("city", back_populates="addressCity")
    # localColName = relationship("childTableClass", back_populates="childColumn")

class city(Base):
    __tablename__ = "cityTbl"
    cityTblid = Column("cityTblId", Integer, primary_key=True)
    cityName = Column("cityName", String)
    addressCity = relationship("address")
    #localColumnName = relationship("TargetParentClass")
    # __table_args__ = (Un)
print("Creating DB engine")
engine = create_engine(sqlDb)
print("Binding endinge to session")
DBSession = sessionmaker(bind=engine)
print("creating session")
session = DBSession()
print("Adding customer")
ed_customer = customer(customerName="Ed_test", addresses=address(addressName="123 Test Ave", cities=city(cityName = "TestCity")))
session.add(ed_customer)
print("Commiting customer changes")
session.commit()