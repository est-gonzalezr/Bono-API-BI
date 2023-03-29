from pydantic import BaseModel

class DataModel(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    year: float
    km_driven: float
    seats: float
    max_power: float
    fuel: str
    transmission: str


#Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns(self):
        return ["year","km_driven", "seats", "max_power", "fuel", "transmission"]
