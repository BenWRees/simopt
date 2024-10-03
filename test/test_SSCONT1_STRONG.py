import unittest
import math

from simopt.experiment_base import ProblemSolver, post_normalize

# Note: Tests have inherent randomness and may vary **slightly** between
#       runs/systems. To make sure these tsts still work, assertAlmostEqual
#       is used instead of assertEqual.
#       Some attributes, such as the lengths of lists, are still checked
#       with assertEqual as these should not change between runs.

class test_SSCONT1_STRONG(unittest.TestCase):
    def setUp(self):
        # Expected values
        self.expected_problem_name = "SSCONT-1"
        self.expected_solver_name = "STRONG"
        self.expected_all_recommended_xs = "[[(600, 600), (598.243699461503, 599.043230216575), (598.243699461503, 599.043230216575)], [(600, 600), (598.7521421680161, 598.4370378023904), (597.3640786416003, 596.9971490404945), (596.3130441598862, 595.2955831386711), (594.9489777351401, 593.8329407391486), (594.9489777351401, 593.8329407391486)], [(600, 600), (599.4398801887686, 601.9199650510012), (598.101035671903, 603.4057292852116), (597.0268165347238, 601.7187043164224), (595.6746228097004, 603.1923298217688), (595.6746228097004, 603.1923298217688)], [(600, 600), (599.7016800054165, 598.0223738521066), (598.1679561437434, 596.7387496406902), (596.4272388477053, 597.7235863252598), (596.4272388477053, 597.7235863252598)], [(600, 600), (599.3882031580599, 598.095871689147), (598.912029045398, 596.1533840535542), (597.23135269916, 595.0692590519072), (595.4788759256611, 596.0330148084904), (593.7606803687875, 595.0093918171799), (592.6305304420773, 596.6594709495723), (590.63168463828, 596.7274080722831), (588.6329786538964, 596.7993412912363), (588.6329786538964, 596.7993412912363)], [(600, 600), (600, 600)], [(600, 600), (599.1794741231304, 598.1760654382936), (598.6704496934138, 600.1102045568556), (597.5309568154792, 598.4665634825526), (595.9056493842735, 597.3010697013304), (595.0827810483021, 599.1239486610136), (593.647475082212, 597.7311468862781), (593.647475082212, 597.7311468862781)], [(600, 600), (598.4099763663844, 598.7868121149039), (596.6414246688615, 599.7207419266075), (595.0571474436849, 598.5000594642993), (593.2954279746002, 597.5533047377476), (592.6083943708315, 595.6750116756051), (592.6083943708315, 595.6750116756051)], [(600, 600), (598.4089358359453, 598.7881770649716), (596.9708599037266, 597.3982354787265), (595.3674289609223, 596.2028237456161), (593.7066741059846, 595.0884210752134), (592.030555693024, 593.9972624864948), (590.0317379066273, 594.0660189900742), (588.0383463704653, 594.2284695473454), (586.0768256423025, 593.8380384612524), (586.0768256423025, 593.8380384612524)], [(600, 600), (598.1416415794554, 599.2607409244446), (596.667213812626, 597.9094220278097), (594.6754744772925, 597.7278334266114), (592.8082759325749, 597.0111957255344), (590.8640326926377, 596.5422413312076), (588.9454118491695, 595.9775342752843), (586.9465454802103, 596.0448636079982), (584.9499578851768, 595.9280818655966), (582.9800360340884, 595.5825273827661), (582.9800360340884, 595.5825273827661)], [(600, 600), (598.0845228274608, 600.5752801069749), (596.7128465858013, 602.0307883646179), (594.8372130311383, 601.336527053135), (592.9129684905145, 600.7912904550609), (591.6167172102138, 599.2682236087542), (591.6167172102138, 599.2682236087542)], [(600, 600), (598.4923149562098, 598.685889727332), (596.5459716008394, 599.1460500722178), (595.5447140624424, 597.4147259137185), (594.6434174845114, 595.6293227439527), (593.6774262480003, 593.878077206918), (592.4326611715995, 592.3126507779303), (592.4326611715995, 592.3126507779303)], [(600, 600), (599.259650053703, 598.1420759011687), (597.3673743869351, 597.4945476771057), (596.0875220556112, 595.9576748749505), (594.809985854327, 594.4188762349693), (592.8279413023331, 594.686269940788), (590.941520664903, 594.0218776028748), (588.9429027024015, 593.9475389186906), (586.9467089171517, 593.8242085828845), (586.9467089171517, 593.8242085828845)], [(600, 600), (599.1131567205473, 601.7926262850046), (598.1124859765897, 603.5242896651647), (596.3335186756129, 604.4382236573409), (594.3772449917163, 604.8541482583128), (594.3772449917163, 604.8541482583128)], [(600, 600), (598.0114276980933, 599.7865048007811), (598.0114276980933, 599.7865048007811)], [(600, 600), (598.5917099207398, 598.5798876619587), (596.8047810592269, 599.4781555751629), (595.6298379514404, 597.85966594175), (593.7346700981271, 598.4986798756886), (592.4277211457012, 596.9847827735866), (591.5170344065162, 595.2041509578896), (589.9152167232555, 594.0065783673577), (588.1469826700728, 593.0720472907395), (588.1469826700728, 593.0720472907395)], [(600, 600), (599.1376847810809, 598.1954467413732), (599.1376847810809, 598.1954467413732)], [(600, 600), (598.1509800615808, 599.237684273199), (596.1509881113127, 599.243358677394), (594.1612306613162, 599.0412070259453), (594.1612306613162, 599.0412070259453)], [(600, 600), (599.1602010802652, 598.184858745328), (598.2326989760305, 596.4129276386988), (597.1866138741443, 594.7083144979505), (596.5375020913059, 592.8165814582801), (596.5375020913059, 592.8165814582801)], [(600, 600), (598.0205637292946, 600.2860630179107), (596.0605400843847, 599.8881845045503), (594.064248442076, 600.0099206078577), (592.0642767171793, 599.9992857776399), (592.0642767171793, 599.9992857776399)], [(600, 600), (598.4155460011777, 601.2204530001667), (596.4309822154833, 601.4684582027536), (594.4481132268681, 601.7296680302645), (593.0474500911862, 603.1573034110049), (591.4101093826587, 604.305830904885), (589.4717723873468, 604.7986287261114), (587.7102486963149, 605.7457476658086), (587.7102486963149, 605.7457476658086)], [(600, 600), (599.3155912709664, 598.1207489013912), (597.9934511487205, 596.620100542465), (597.9934511487205, 596.620100542465)], [(600, 600), (598.9133192979835, 598.3209749698515), (598.2688166774637, 596.4276666557771), (596.7057847800945, 597.6754371826796), (596.7057847800945, 597.6754371826796)], [(600, 600), (598.3918622864361, 598.8109276328947), (596.6315099389471, 599.760221903756), (595.1494721562551, 598.4172535457247), (595.1494721562551, 598.4172535457247)]]"
        self.expected_all_intermediate_budgets = "[[10, 60, 1000], [10, 60, 115, 550, 640, 1000], [10, 60, 115, 175, 550, 1000], [10, 60, 115, 240, 1000], [10, 60, 115, 175, 240, 310, 385, 465, 550, 1000], [10, 1000], [10, 60, 115, 175, 240, 385, 465, 1000], [10, 60, 175, 385, 465, 940, 1000], [10, 60, 115, 175, 240, 310, 385, 465, 640, 1000], [10, 60, 115, 175, 240, 310, 385, 550, 735, 835, 1000], [10, 60, 175, 240, 550, 640, 1000], [10, 60, 115, 310, 465, 550, 940, 1000], [10, 60, 115, 175, 465, 550, 640, 835, 940, 1000], [10, 60, 385, 465, 550, 1000], [10, 60, 1000], [10, 60, 115, 175, 240, 310, 385, 465, 550, 1000], [10, 60, 1000], [10, 60, 115, 175, 1000], [10, 60, 385, 550, 640, 1000], [10, 60, 240, 640, 735, 1000], [10, 60, 115, 175, 550, 640, 735, 835, 1000], [10, 60, 115, 1000], [10, 60, 175, 310, 1000], [10, 175, 385, 465, 1000]]"
        self.expected_all_est_objectives = "[[618.5809976153716, 617.681414579216, 617.681414579216], [619.371245290233, 618.0089737925229, 616.9013835756483, 615.4140252291478, 614.179301173641, 614.179301173641], [620.2040298994102, 620.6859097978055, 621.4573228291982, 619.6928950910417, 620.5212274757799, 620.5212274757799], [620.3929887875448, 619.3361952627671, 618.0164727376502, 617.7125448935677, 617.7125448935677], [617.140803174291, 616.2673924879126, 615.8443322420237, 615.2356096929472, 614.3456740584402, 613.5884654820031, 612.987448023437, 612.1516136240574, 611.2003689529062, 611.2003689529062], [617.6250759903628, 617.6250759903628], [622.8299886318688, 622.5380857135009, 622.1911823969084, 621.8413452582026, 621.0552828187981, 620.3140570938123, 620.1786938899882, 620.1786938899882], [617.1638109984892, 616.4925485876045, 615.5523232876876, 614.8797709986508, 613.768141057626, 614.6452669754458, 614.6452669754458], [625.4509909440814, 624.1954091655041, 622.6408223045593, 622.1592775871597, 620.9065555537599, 619.8721338145766, 619.0267204112898, 618.2443905555721, 617.2041049666219, 617.2041049666219], [616.3517529689802, 614.9427926225667, 614.2710403753969, 613.2604275356047, 612.3510424143137, 610.8738965271837, 610.2624806442021, 609.4425827622551, 608.5932339646369, 607.5622301510275, 607.5622301510275], [620.885724515664, 619.7561359689448, 619.2618957355273, 618.5307613363816, 617.5470353014966, 616.7980950716151, 616.7980950716151], [618.8018614763591, 617.195795229819, 617.4640664330827, 615.1494013121154, 612.7713155558853, 611.4567603575118, 610.8030286585318, 610.8030286585318], [619.9876951863847, 619.7001864999235, 618.6355951605684, 617.1031836722602, 615.6269053908736, 614.7128775700157, 613.7619444355242, 612.9541432910837, 612.029869615391, 612.029869615391], [620.5752955225382, 620.1461227475561, 620.5784726832604, 619.7853976021572, 619.2830465374145, 619.2830465374145], [624.383736629565, 623.3877572835046, 623.3877572835046], [621.6851306868389, 620.7678241488641, 620.2274294348496, 619.4774897680247, 618.9925385801879, 617.8010685656436, 616.8760525932308, 615.5336330505016, 614.3967389181754, 614.3967389181754], [619.7087176057688, 618.7001778270516, 618.7001778270516], [623.808754566687, 622.1747638446441, 621.2674704138929, 620.3200792553803, 620.3200792553803], [618.1205002744614, 617.8546186017838, 616.4593153879028, 616.2754760423658, 615.826817559924, 615.826817559924], [621.6456203016469, 620.7907881855399, 619.8839280139259, 619.0197482886304, 618.1393776971213, 618.1393776971213], [617.7541201292993, 618.0911130495334, 617.2613468552839, 616.2319242697208, 616.0455875500301, 614.8043597778106, 614.57545211384, 614.1712190608864, 614.1712190608864], [626.0524700155847, 624.9683316203887, 623.9249452109043, 623.9249452109043], [616.5518744333754, 615.9121620810129, 614.5436865811713, 614.7562700748224, 614.7562700748224], [619.165760431194, 618.1453545905769, 617.3816757217435, 616.5642735889041, 616.5642735889041]]"
        self.expected_objective_curves = "[([10, 60, 1000], [624.4131899421741, 617.681414579216, 617.681414579216]), ([10, 60, 115, 550, 640, 1000], [624.4131899421741, 618.0089737925229, 616.9013835756483, 615.4140252291478, 614.179301173641, 614.179301173641]), ([10, 60, 115, 175, 550, 1000], [624.4131899421741, 620.6859097978055, 621.4573228291982, 619.6928950910417, 620.5212274757799, 620.5212274757799]), ([10, 60, 115, 240, 1000], [624.4131899421741, 619.3361952627671, 618.0164727376502, 617.7125448935677, 617.7125448935677]), ([10, 60, 115, 175, 240, 310, 385, 465, 550, 1000], [624.4131899421741, 616.2673924879126, 615.8443322420237, 615.2356096929472, 614.3456740584402, 613.5884654820031, 612.987448023437, 612.1516136240574, 611.2003689529062, 611.2003689529062]), ([10, 1000], [624.4131899421741, 624.4131899421741]), ([10, 60, 115, 175, 240, 385, 465, 1000], [624.4131899421741, 622.5380857135009, 622.1911823969084, 621.8413452582026, 621.0552828187981, 620.3140570938123, 620.1786938899882, 620.1786938899882]), ([10, 60, 175, 385, 465, 940, 1000], [624.4131899421741, 616.4925485876045, 615.5523232876876, 614.8797709986508, 613.768141057626, 614.6452669754458, 614.6452669754458]), ([10, 60, 115, 175, 240, 310, 385, 465, 640, 1000], [624.4131899421741, 624.1954091655041, 622.6408223045593, 622.1592775871597, 620.9065555537599, 619.8721338145766, 619.0267204112898, 618.2443905555721, 617.2041049666219, 617.2041049666219]), ([10, 60, 115, 175, 240, 310, 385, 550, 735, 835, 1000], [624.4131899421741, 614.9427926225667, 614.2710403753969, 613.2604275356047, 612.3510424143137, 610.8738965271837, 610.2624806442021, 609.4425827622551, 608.5932339646369, 616.9362325231637, 616.9362325231637]), ([10, 60, 175, 240, 550, 640, 1000], [624.4131899421741, 619.7561359689448, 619.2618957355273, 618.5307613363816, 617.5470353014966, 616.7980950716151, 616.7980950716151]), ([10, 60, 115, 310, 465, 550, 940, 1000], [624.4131899421741, 617.195795229819, 617.4640664330827, 615.1494013121154, 612.7713155558853, 611.4567603575118, 610.8030286585318, 610.8030286585318]), ([10, 60, 115, 175, 465, 550, 640, 835, 940, 1000], [624.4131899421741, 619.7001864999235, 618.6355951605684, 617.1031836722602, 615.6269053908736, 614.7128775700157, 613.7619444355242, 612.9541432910837, 612.029869615391, 612.029869615391]), ([10, 60, 385, 465, 550, 1000], [624.4131899421741, 620.1461227475561, 620.5784726832604, 619.7853976021572, 619.2830465374145, 619.2830465374145]), ([10, 60, 1000], [624.4131899421741, 623.3877572835046, 623.3877572835046]), ([10, 60, 115, 175, 240, 310, 385, 465, 550, 1000], [624.4131899421741, 620.7678241488641, 620.2274294348496, 619.4774897680247, 618.9925385801879, 617.8010685656436, 616.8760525932308, 615.5336330505016, 614.3967389181754, 614.3967389181754]), ([10, 60, 1000], [624.4131899421741, 618.7001778270516, 618.7001778270516]), ([10, 60, 115, 175, 1000], [624.4131899421741, 622.1747638446441, 621.2674704138929, 620.3200792553803, 620.3200792553803]), ([10, 60, 385, 550, 640, 1000], [624.4131899421741, 617.8546186017838, 616.4593153879028, 616.2754760423658, 615.826817559924, 615.826817559924]), ([10, 60, 240, 640, 735, 1000], [624.4131899421741, 620.7907881855399, 619.8839280139259, 619.0197482886304, 618.1393776971213, 618.1393776971213]), ([10, 60, 115, 175, 550, 640, 735, 835, 1000], [624.4131899421741, 618.0911130495334, 617.2613468552839, 616.2319242697208, 616.0455875500301, 614.8043597778106, 614.57545211384, 614.1712190608864, 614.1712190608864]), ([10, 60, 115, 1000], [624.4131899421741, 624.9683316203887, 623.9249452109043, 623.9249452109043]), ([10, 60, 175, 310, 1000], [624.4131899421741, 615.9121620810129, 614.5436865811713, 614.7562700748224, 614.7562700748224]), ([10, 175, 385, 465, 1000], [624.4131899421741, 618.1453545905769, 617.3816757217435, 616.5642735889041, 616.5642735889041])]"
        self.expected_progress_curves = "[([0.01, 0.06, 1.0], [1.0, 0.09966380899237617, 0.09966380899237617]), ([0.01, 0.06, 0.115, 0.55, 0.64, 1.0], [1.0, 0.14347296757792355, -0.004660846058416508, -0.20358646020179147, -0.36872369267652794, -0.36872369267652794]), ([0.01, 0.06, 0.115, 0.175, 0.55, 1.0], [1.0, 0.5014977436019833, 0.604669794499504, 0.3686877446792896, 0.47947243132631817, 0.47947243132631817]), ([0.01, 0.06, 0.115, 0.24, 1.0], [1.0, 0.3209811966430954, 0.1444759083073075, 0.10382730927827258, 0.10382730927827258]), ([0.01, 0.06, 0.115, 0.175, 0.24, 0.31, 0.385, 0.465, 0.55, 1.0], [1.0, -0.08945350331278969, -0.14603537507967435, -0.22744851079298756, -0.3464722773647051, -0.44774456420585, -0.5281271884318558, -0.6399152263380803, -0.7671387235232682, -0.7671387235232682]), ([0.01, 1.0], [1.0, 1.0]), ([0.01, 0.06, 0.115, 0.175, 0.24, 0.385, 0.465, 1.0], [1.0, 0.7492156068850044, 0.7028192858747397, 0.6560305830507418, 0.5508992581877782, 0.4517645856936905, 0.43366053664829307, 0.43366053664829307]), ([0.01, 0.06, 0.175, 0.385, 0.465, 0.94, 1.0], [1.0, -0.05934017150226247, -0.1850898912380323, -0.2750398871182891, -0.4237139905976479, -0.30640344987026386, -0.30640344987026386]), ([0.01, 0.06, 0.115, 0.175, 0.24, 0.31, 0.385, 0.465, 0.64, 1.0], [1.0, 0.9708730751740967, 0.7629560343478067, 0.6985522012892994, 0.531007842909673, 0.3926598918362604, 0.2795907173165165, 0.17495860402820748, 0.03582639681445608, 0.03582639681445608]), ([0.01, 0.06, 0.115, 0.175, 0.24, 0.31, 0.385, 0.55, 0.735, 0.835, 1.0], [1.0, -0.2666111078188804, -0.3564541027063284, -0.49161775058570273, -0.6132427740182014, -0.8108025305274958, -0.892575884141492, -1.002232504608845, -1.1158280154591262, 0.0, 0.0]), ([0.01, 0.06, 0.175, 0.24, 0.55, 0.64, 1.0], [1.0, 0.37714584793695083, 0.3110440627160183, 0.2132590469438547, 0.08169135439771179, -0.018475088703508426, -0.018475088703508426]), ([0.01, 0.06, 0.115, 0.31, 0.465, 0.55, 0.94, 1.0], [1.0, 0.034715017367285536, 0.07059474600952588, -0.23897838531287735, -0.5570336614046941, -0.7328478495437293, -0.8202806998790603, -0.8202806998790603]), ([0.01, 0.06, 0.115, 0.175, 0.465, 0.55, 0.64, 0.835, 0.94, 1.0], [1.0, 0.36966292863088346, 0.2272799672610239, 0.02232875483175774, -0.1751149644053108, -0.2973609221706892, -0.42454275312156653, -0.5325815046044449, -0.6561977864549594, -0.6561977864549594]), ([0.01, 0.06, 0.385, 0.465, 0.55, 1.0], [1.0, 0.4293043339034107, 0.4871286481900044, 0.3810594228809516, 0.31387286067512016, 0.31387286067512016]), ([0.01, 0.06, 1.0], [1.0, 0.8628542866832032, 0.8628542866832032]), ([0.01, 0.06, 0.115, 0.175, 0.24, 0.31, 0.385, 0.465, 0.55, 1.0], [1.0, 0.5124533163661535, 0.44017863513812844, 0.33987852310083166, 0.2750190942369168, 0.11566684066985432, -0.00804871909258241, -0.1875896028370924, -0.3396426464234634, -0.3396426464234634]), ([0.01, 0.06, 1.0], [1.0, 0.2359175270148007, 0.2359175270148007]), ([0.01, 0.06, 0.115, 0.175, 1.0], [1.0, 0.7006233990528379, 0.5792781271853804, 0.4525700151258149, 0.4525700151258149]), ([0.01, 0.06, 0.385, 0.55, 0.64, 1.0], [1.0, 0.12282884964479777, -0.06378492059460264, -0.08837237445245964, -0.14837786295517397, -0.14837786295517397]), ([0.01, 0.06, 0.24, 0.64, 0.735, 1.0], [1.0, 0.5155246240370309, 0.3942372980843317, 0.27865823605859874, 0.1609137389091714, 0.1609137389091714]), ([0.01, 0.06, 0.115, 0.175, 0.55, 0.64, 0.735, 0.835, 1.0], [1.0, 0.15445862021809043, 0.04348216980528258, -0.0941971732582164, -0.1191186365284217, -0.2851257036629157, -0.31574078559298324, -0.36980462871797193, -0.36980462871797193]), ([0.01, 0.06, 0.115, 1.0], [1.0, 1.0742470027718896, 0.9347000786672386, 0.9347000786672386]), ([0.01, 0.06, 0.175, 0.31, 1.0], [1.0, -0.1369635247015118, -0.3199892426709853, -0.2915574245211862, -0.2915574245211862]), ([0.01, 0.175, 0.385, 0.465, 1.0], [1.0, 0.16171311399192517, 0.05957546280085378, -0.04974736559469913, -0.04974736559469913])]"

        # Convert the expected values from string to their actual types
        self.expected_all_recommended_xs = eval(self.expected_all_recommended_xs, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_intermediate_budgets = eval(self.expected_all_intermediate_budgets, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_all_est_objectives = eval(self.expected_all_est_objectives, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_objective_curves = eval(self.expected_objective_curves, {'nan': float('nan'), 'inf': float('inf')})
        self.expected_progress_curves = eval(self.expected_progress_curves, {'nan': float('nan'), 'inf': float('inf')})
        
        # Number of macro-replications and post-replications
        self.num_macroreps = 24
        self.num_postreps = 200

        # Setup the solver and experiment
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.assertEqual(self.myexperiment.solver.name, self.expected_solver_name, "Solver name does not match (expected: " + self.expected_solver_name + ", actual: " + self.myexperiment.solver.name + ")")
        self.assertEqual(self.myexperiment.problem.name, self.expected_problem_name, "Problem name does not match (expected: " + self.expected_problem_name + ", actual: " + self.myexperiment.problem.name + ")")

    def test_run(self):
        # Check actual run results against expected
        self.myexperiment.run(n_macroreps=self.num_macroreps)
        self.assertEqual(self.myexperiment.n_macroreps, self.num_macroreps, "Number of macro-replications for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep]), len(self.expected_all_recommended_xs[mrep]), "Length of recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list of recommended solutions
            for list in range(len(self.myexperiment.all_recommended_xs[mrep])):
                # Check to make sure the tuples are the same length
                self.assertEqual(len(self.myexperiment.all_recommended_xs[mrep][list]), len(self.expected_all_recommended_xs[mrep][list]), "Recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
                # For each tuple of recommended solutions
                for tuple in range(len(self.myexperiment.all_recommended_xs[mrep][list])):
                    self.assertAlmostEqual(self.myexperiment.all_recommended_xs[mrep][list][tuple], self.expected_all_recommended_xs[mrep][list][tuple], 5, "Recommended solutions for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + " and tuple " + str(tuple) + ".")
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_intermediate_budgets[mrep]), len(self.expected_all_intermediate_budgets[mrep]), "Length of intermediate budgets for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list of intermediate budgets
            for list in range(len(self.myexperiment.all_intermediate_budgets[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_intermediate_budgets[mrep][list], self.expected_all_intermediate_budgets[mrep][list], 5, "Intermediate budgets for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")
            
    def test_post_replicate(self):
        # Simulate results from the run method
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = self.expected_all_intermediate_budgets

        # Check actual post-replication results against expected
        self.myexperiment.post_replicate(n_postreps=self.num_postreps)
        self.assertEqual(self.myexperiment.n_postreps, self.num_postreps, "Number of post-replications for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
        # For each macroreplication
        for mrep in range(self.num_macroreps):
            # Check to make sure the list lengths are the same
            self.assertEqual(len(self.myexperiment.all_est_objectives[mrep]), len(self.expected_all_est_objectives[mrep]), "Estimated objectives for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
            # For each list in the estimated objectives
            for list in range(len(self.myexperiment.all_est_objectives[mrep])):
                # Check the values in the list
                self.assertAlmostEqual(self.myexperiment.all_est_objectives[mrep][list], self.expected_all_est_objectives[mrep][list], 5, "Estimated objectives for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(list) + ".")

    def test_post_normalize(self):
        # Simulate results from the post_replicate method
        self.myexperiment = ProblemSolver(self.expected_solver_name, self.expected_problem_name)
        self.myexperiment.n_macroreps = self.num_macroreps
        self.myexperiment.n_postreps = self.num_postreps
        self.myexperiment.all_recommended_xs = self.expected_all_recommended_xs
        self.myexperiment.all_intermediate_budgets = self.expected_all_intermediate_budgets
        self.myexperiment.all_est_objectives = self.expected_all_est_objectives
        self.myexperiment.has_run = True
        self.myexperiment.has_postreplicated= True
        # Check actual post-normalization results against expected
        post_normalize([self.myexperiment], n_postreps_init_opt=self.num_postreps)

        # Loop through each curve object and convert it into a tuple
        for i in range(len(self.myexperiment.objective_curves)):
            self.myexperiment.objective_curves[i] = (self.myexperiment.objective_curves[i].x_vals, self.myexperiment.objective_curves[i].y_vals)
        for i in range(len(self.myexperiment.progress_curves)):
            self.myexperiment.progress_curves[i] = (self.myexperiment.progress_curves[i].x_vals, self.myexperiment.progress_curves[i].y_vals)

        for mrep in range(self.num_macroreps):
            # Check to make sure the same number of objective curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(len(self.myexperiment.objective_curves[mrep]), len(self.expected_objective_curves[mrep]), "Number of objective curves for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.objective_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][0]), len(self.expected_objective_curves[mrep][0]), "Length of X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.objective_curves[mrep][1]), len(self.expected_objective_curves[mrep][1]), "Length of Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.objective_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(self.expected_objective_curves[mrep][0][x_index]), "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][0][x_index], self.expected_objective_curves[mrep][0][x_index], 5, "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.objective_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.objective_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(self.expected_objective_curves[mrep][1][y_index]), "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.objective_curves[mrep][1][y_index], self.expected_objective_curves[mrep][1][y_index], 5, "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
            
            # Check to make sure the same number of progress curves are present
            # This should probably always be 2 (x and y)
            self.assertEqual(len(self.myexperiment.progress_curves[mrep]), len(self.expected_progress_curves[mrep]), "Number of progress curves for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " does not match.")
            # Make sure that curves are only checked if they exist
            if (len(self.myexperiment.progress_curves[mrep]) > 0):
                # Make sure the lengths of the X and Y values are the same
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][0]), len(self.expected_progress_curves[mrep][0]), "Length of X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                self.assertEqual(len(self.myexperiment.progress_curves[mrep][1]), len(self.expected_progress_curves[mrep][1]), "Length of Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match.")
                # Check X (0) and Y (1) values
                for x_index in range(len(self.myexperiment.progress_curves[mrep][0])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][0][x_index])):
                        self.assertTrue(math.isnan(self.expected_progress_curves[mrep][0][x_index]), "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][0][x_index], self.expected_progress_curves[mrep][0][x_index], 5, "X values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(x_index) + ".")
                for y_index in range(len(self.myexperiment.progress_curves[mrep][1])):
                    # If the value is NaN, make sure we're expecting NaN
                    if (math.isnan(self.myexperiment.progress_curves[mrep][1][y_index])):
                        self.assertTrue(math.isnan(self.expected_progress_curves[mrep][1][y_index]), "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")
                    # Otherwise, check the value normally
                    else:
                        self.assertAlmostEqual(self.myexperiment.progress_curves[mrep][1][y_index], self.expected_progress_curves[mrep][1][y_index], 5, "Y values for problem " + self.expected_problem_name + " and solver " + self.expected_solver_name + " do not match at mrep " + str(mrep) + " and index " + str(y_index) + ".")      
