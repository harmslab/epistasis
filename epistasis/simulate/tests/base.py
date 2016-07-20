from nose import tools

class testBaseClass:

    def setUp(self):
        self.wildtype = "AAAA"
        self.mutations = {
            0 : ["A", "V"],
            1 : ["A", "V"],
            2 : ["A", "V"],
            3 : ["A", "V"],
        }
        self.order = 4
