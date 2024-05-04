## Food recommender

Uses LLM knowledge + Qdrant's Recommendation feature to help you discover new items from a menu that are worth trying!

How it works:
- You give it names of dishes you like/dislike.
- It will generate details of the dish like how it tastes, feels, etc.
- You can build its knowledge base like this over time.
- You can pass it a list of items at any point to pick the ones that you're very likely to love (based on Qdrant's recommendation API)


### Run:

```sh
git clone github.com/KShivendu/rag-cookbook; cd rag-cookbook
pip install -e .
python -i food_recommender/main.py
```
