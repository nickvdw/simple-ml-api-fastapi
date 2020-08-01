from fastapi import FastAPI
from ml import nlp
from pydantic import BaseModel
from typing import List


app = FastAPI()


class Article(BaseModel):
    content: str
    comments: List[str] = []


@app.post("/article/")
def analyze_article(articles: List[Article]):
    """
    Analyze a list of articles by extracting and classifying entities.

    The result *might* contains incorrect extractions and classifications.
    """
    ents = []
    comments = []
    for article in articles:
        for comment in article.comments:
            comments.append(comment.upper())
        doc = nlp(article.content)
        for ent in doc.ents:
            ents.append({"text": ent.text, "label": ent.label_})
        return {"ents": ents, "comments": comments}


@app.get("/")
def main():
    """
    Test method.
    """
    return {"message": "Test message"}

