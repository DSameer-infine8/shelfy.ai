from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class DetectedBook(BaseModel):
    id: str = Field(max_length=100, description="Unique identifier for the detected book")
    spine_index: int = Field(ge=0, description="1-based index of the book spine on shelf")
    title: str = Field(max_length=500, description="Cleaned, actual published book title")
    author: str = Field(max_length=500, description="Cleaned, actual published author name")
    raw_ocr_text: str = Field(default="", max_length=2000, description="Original raw OCR text extracted from the spine")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")


class ScanShelfResponse(BaseModel):
    total_spines_detected: int = Field(ge=0, description="Total number of segmented spines detected")
    books: List[DetectedBook] = Field(default_factory=list, description="List of detected and cleaned book records")
    annotated_image: Optional[str] = Field(default=None, description="Base64 encoded image with YOLO spine segmentation polygons drawn")
    polygons: Optional[List[List[List[float]]]] = Field(default=None, description="Normalized (0.0 to 1.0) polygon coordinates for each detected spine")


class MCQAnswer(BaseModel):
    question: str = Field(max_length=500, description="Question identifier or title")
    answer: Union[str, List[str]] = Field(description="User selected answer or list of answers")


class StructuredMCQAnswers(BaseModel):
    looking_for: Union[str, List[str]] = Field(default_factory=list, description="What user is looking for in next book")
    core_payoff: Optional[str] = Field(default=None, max_length=500, description="Core payoff user wants to walk away with on final page")
    topics: List[str] = Field(default_factory=list, description="Topics of genuine interest (up to 5)")
    reading_experience: str = Field(default="", max_length=500, description="Preferred reading experience")
    life_goal: str = Field(default="", max_length=500, description="Current life goal/focus book should assist with")
    life_goal_detail: Optional[str] = Field(default=None, max_length=1000, description="Optional context for life goal")
    dealbreakers: Optional[str] = Field(default=None, max_length=1000, description="Themes or styles user wants to strictly avoid")
    additional_notes: Optional[str] = Field(default=None, max_length=2000, description="Open-ended ideal book description or notes")


class RecommendationRequest(BaseModel):
    books: List[DetectedBook] = Field(max_length=100, description="List of detected books from the bookshelf (max 100)")
    mcq_answers: Union[StructuredMCQAnswers, Dict[str, Any], List[MCQAnswer]] = Field(
        description="User multiple-choice questionnaire answers"
    )


class BookRecommendation(BaseModel):
    rank: int = Field(ge=1, le=3, description="Rank from 1 to 3")
    title: str = Field(max_length=500, description="Title of the recommended book")
    author: str = Field(max_length=500, description="Author of the recommended book")
    match_reason: str = Field(max_length=2000, description="Concise personalized rationale for the recommendation")
    genre: Optional[str] = Field(default=None, max_length=200, description="Genre or category of the book")
    cover_url: Optional[str] = Field(default=None, max_length=2000, description="URL to book cover image if available")
    isbn_13: Optional[str] = Field(default=None, max_length=50, description="ISBN-13 of the recommended edition")
    publisher: Optional[str] = Field(default=None, max_length=300, description="Publisher name")
    published_date: Optional[str] = Field(default=None, max_length=50, description="Publication date")


class RecommendationResponse(BaseModel):
    recommendations: List[BookRecommendation] = Field(
        description="Top 3 recommended books ordered by rank"
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = True
    yolo_loaded: bool = True
    ocr_loaded: bool = True
    version: str = "1.0.0"


class SavedBookItem(BaseModel):
    id: str = Field(max_length=100, description="Unique book identifier")
    title: str = Field(max_length=500, description="Book title")
    author: str = Field(max_length=500, description="Book author")
    cover_url: Optional[str] = Field(default=None, max_length=2000, description="Book cover URL")
    match_score: Optional[int] = Field(default=95, ge=0, le=100, description="Match score percentage")
    category: Optional[str] = Field(default=None, max_length=200, description="Genre or category")
    reading_status: str = Field(default="unread", max_length=50, description="Status: unread, reading, completed")
    source_table: str = Field(default="detected_books", max_length=50, description="Source table: detected_books, recommendations, manual")
    is_saved: bool = Field(default=True, description="Whether book is saved in favorite library")
    user_id: Optional[str] = Field(default=None, max_length=100, description="User UUID")
    scan_id: Optional[str] = Field(default=None, max_length=100, description="Scan ID if originated from a shelf scan")
    created_at: Optional[str] = Field(default=None, max_length=100, description="Creation timestamp")


class SavedBooksResponse(BaseModel):
    total: int = Field(ge=0, description="Total count of saved books")
    books: List[SavedBookItem] = Field(default_factory=list, description="List of saved book items")


class ToggleSaveBookRequest(BaseModel):
    book_id: str = Field(max_length=100, description="Unique ID of the book")
    is_saved: bool = Field(default=True, description="Target saved status")
    source_table: Optional[str] = Field(default="detected_books", max_length=50, description="Source table: detected_books or recommendations")
    title: Optional[str] = Field(default=None, max_length=500, description="Optional book title for manual creation")
    author: Optional[str] = Field(default=None, max_length=500, description="Optional book author for manual creation")
    cover_url: Optional[str] = Field(default=None, max_length=2000, description="Optional book cover image URL")
    user_id: Optional[str] = Field(default=None, max_length=100, description="Optional user ID")
    reading_status: Optional[str] = Field(default="unread", max_length=50, description="Optional reading status")


class SegmentOverlayRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, description="Public URL of the bookshelf image to segment")


class SegmentOverlayResponse(BaseModel):
    annotated_image: Optional[str] = Field(default=None, description="Base64 encoded JPEG data URI with segmentation polygons & badges")
    polygons: List[List[List[float]]] = Field(default_factory=list, description="Normalized [0..1] polygon coordinates for responsive UI")
    total_spines: int = Field(default=0, description="Total detected spines")



