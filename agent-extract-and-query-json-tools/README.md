# Overview:

This is an agent that can use multiple tools and query JSON data with SQL (using SQLite).

It also has a special feature to extract new fields from existing data to make it easier to query with SQL (or payload filtering in case of vector DBs)

This means if you have
```jsonc
// menu.json
[
    {
        "name": "Paneer biriyani",
        "price": 22.9,
        "section": "Rice dish"
    },
    {
        "name": "Chicken fried rice",
        "price": 14.9,
        "section": "Rice dish"
    },
    ...
]
```

And query is `How many veg items are present`. It will extract/infer a new field calleed `is_veg` of type `bool` based on existing fields like `name`

```jsonc
// updated-menu.json
[
    {
        "name": "Paneer biriyani",
        "price": 22.9,
        "section": "Rice dish",
        "is_veg": true,
    },
    {
        "name": "Chicken fried rice",
        "price": 14.9,
        "section": "Rice dish"
        "is_veg": false,
    },
    ...
]
```

Now the agent will dump this into SQLite and proceed with answering the query.
