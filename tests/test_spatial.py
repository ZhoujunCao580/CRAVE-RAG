from softdoc.models import ElementType
from softdoc.spatial import SpatialNavigator


def test_caption_is_below_figure(parsed_document) -> None:
    navigator = SpatialNavigator(parsed_document)
    figure = next(
        element
        for element in parsed_document.elements
        if element.element_type == ElementType.FIGURE and element.page_number == 3
    )
    below = navigator.get_elements_below(figure.element_id)
    assert below
    assert below[0].element_type == ElementType.CAPTION


def test_spatial_navigation_and_adjacent_pages(parsed_document) -> None:
    navigator = SpatialNavigator(parsed_document)
    figure = next(element for element in parsed_document.elements if element.element_type == ElementType.FIGURE)
    nearby = navigator.get_nearby_elements(figure.element_id, max_distance=0.1)
    assert any(element.element_type == ElementType.CAPTION for element in nearby)
    assert [page.page_number for page in navigator.get_adjacent_pages(parsed_document.pages[1].page_id)] == [1, 3]

