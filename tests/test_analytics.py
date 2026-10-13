from roadwatch.analytics import crossed_line, signed_line_side, summarize_crossings

def test_signed_line_side():
    assert signed_line_side((5,5),(0,0),(10,0)) > 0
    assert signed_line_side((5,-5),(0,0),(10,0)) < 0

def test_crossing_directions():
    start,end=(0,0),(10,0)
    assert crossed_line((5,-5),(5,5),start,end)=="negative_to_positive"
    assert crossed_line((5,5),(5,-5),start,end)=="positive_to_negative"

def test_no_crossing():
    assert crossed_line((2,-5),(8,-3),(0,0),(10,0)) is None

def test_summary_aggregation():
    events=[{"class_name":"car","direction":"positive_to_negative"},{"class_name":"car","direction":"positive_to_negative"},{"class_name":"bus","direction":"negative_to_positive"}]
    result=summarize_crossings(events)
    assert {"class_name":"car","direction":"positive_to_negative","count":2} in result
    assert {"class_name":"bus","direction":"negative_to_positive","count":1} in result
