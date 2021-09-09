import json

import fire
from loguru import logger


@logger.catch(reraise=True)
def category_reorganization(
    annotations_file: str,
    cats_to_merge: list,
    final_cat: str,
    output_file: str,
    categories: str = "categories",
    category_id: str = "category_id",
) -> None:
    """This function can be used to reorganize the set of categories of a
    dataset.

    Concretely, it can be use to merge sets of categories. The new annotations
    are stored into the specified output .json file.

    Args:
        annotations_file (str):
        cats_to_merge (list): List of strings of categories to merge.
        final_cat (str): Name of the resulting category.
        output_file (str):
        categories (str): Name of the categories field in the annotations.
        category_id (str): Name of the category ids field in the annotations.

    Returns:
        metadata: Dictionary with the annotations.

    """
    metadata = json.load(open(annotations_file))
    old_cats = [cat["name"] for cat in metadata[categories]]

    # Format restriction: 'name' and 'id' must appear in the annotations file
    cat_ids_to_change = list()
    for cat in cats_to_merge:
        if cat in old_cats:
            for ann_cat in metadata[categories]:
                if ann_cat["name"] == cat:
                    cat_ids_to_change.append(ann_cat["id"])
            old_cats.remove(cat)
            continue
        else:
            raise ValueError(f"{cat} not a class from the dataset.")
    old_cats.append(final_cat)

    # +1 because `categories` indices in COCO format are expected to start at 1
    new_cats = [
        {"id": (category_id + 1), "name": name}
        for category_id, name in enumerate(old_cats)
    ]

    old_to_new = list()
    for new_cat in new_cats:
        for old_cat in metadata[categories]:
            if old_cat["name"] == new_cat["name"]:
                this_old_to_new = {
                    "old_id": old_cat["id"],
                    "new_id": new_cat["id"],
                }
                old_to_new.append(this_old_to_new)

    metadata[categories] = new_cats
    for ann_cat in metadata[categories]:
        if ann_cat["name"] == final_cat:
            new_cat_id = ann_cat["id"]

    # Update category of each annotation
    for ann in metadata["annotations"]:
        if ann[category_id] in cat_ids_to_change:
            ann[category_id] = new_cat_id

    with open(output_file, "w") as f:
        json.dump(metadata, f)

    return metadata


if __name__ == "__main__":
    fire.Fire(category_reorganization)
