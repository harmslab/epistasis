# from nose import tools
#
# from ..mapping import *
#
# def test_mutations_to_sites_1():
#     # Prepare data
#     mutations = {0:["A","V"], 1:["A", "V"]}
#     expected = [[0], [1], [2]]
#     # Test function
#     check = mutations_to_sites(1, mutations)
#     expected = [tuple(x) for x in expected]
#     check = [tuple(x) for x in check]
#     # Run tests
#     tools.assert_equals(set(expected), set(check))
#
# def test_mutations_to_sites_2():
#     # Prepare data
#     mutations = {0:["A","V"], 1:["A", "V"]}
#     expected = [[0], [1], [2], [1,2]]
#     # Test function
#     check = mutations_to_sites(2, mutations)
#     expected = [tuple(x) for x in expected]
#     check = [tuple(x) for x in check]
#     # Run tests
#     tools.assert_equals(set(expected), set(check))
