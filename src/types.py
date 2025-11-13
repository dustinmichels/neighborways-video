from pydantic import BaseModel, Field


class ImgRecord(BaseModel):
    saved_path: str = Field(..., description="Path to the saved image file")
    label: str = Field(..., description="Detected object class")
    track_id: int = Field(..., description="Unique track identifier for the object")
    frame_no: int = Field(..., description="Frame number where the object was detected")
    conf: float = Field(..., description="Confidence score of the detection")
