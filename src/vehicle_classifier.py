"""
Módulo de classificação de veículos (Fase 2 - VMMR):
- Estimativa da cor da lataria (espaço HSV com correção de iluminação)
- Mapeamento de tipo de veículo (Carro, Moto, Caminhão, Ônibus)
- Classificação Zero-Shot de Marca, Modelo e Carroceria via Google SigLIP
  (com Prompt Ensembling da frota brasileira e cache de embeddings de texto em RAM/disco)
"""

import os
import sys
import hashlib
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Mapeamento de classes do modelo COCO para português
MAPA_TIPOS_VEICULO = {
    2: "Carro",
    3: "Moto",
    5: "Ônibus",
    7: "Caminhão",
}

# Catálogo de Prompt Ensembling da frota brasileira (Marca/Modelo -> Categoria + Prompts visuais)
# Cada modelo possui múltiplas descrições visuais (frente, traseira, silhueta) cujos vetores
# matemáticos são combinados (média normalizada) uma única vez na inicialização.
CATALOGO_FROTA_BR = {
    "Volkswagen Fox": {
        "categoria": "Hatchback",
        "prompts": [
            "rear view of a Volkswagen Fox hatchback car in Brazil",
            "rear view of a Volkswagen Fox with oval taillights and high roof",
            "rear view of a silver Volkswagen Fox car with circular taillight reflectors and centered VW logo",
            "rear view of a Volkswagen Fox tall boy hatchback",
            "front view of a Volkswagen Fox compact hatchback car showing front grille and headlights",
            "front angle of a Volkswagen Fox car on a street",
        ],
    },
    "Volkswagen Gol": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Volkswagen Gol subcompact hatchback car with low front hood, horizontal grille and VW badge",
            "front angle of a standard low-roof Volkswagen Gol hatchback passenger car in Brazil",
            "rear view of a Volkswagen Gol compact hatchback car with low roofline and squarish horizontal taillights flanking the tailgate",
            "rear angle of a Volkswagen Gol economy hatchback showing compact rear bumper and VW logo",
            "a photo of a Volkswagen Gol subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Volkswagen Polo": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a modern Volkswagen Polo compact hatchback car with sharp angular headlights, slim front grille and VW emblem",
            "front angle of a modern Volkswagen Polo hatchback showing sculpted hood creases and sporty front bumper",
            "rear view of a Volkswagen Polo hatchback car with sharp geometric wraparound taillights",
            "rear angle of a modern Volkswagen Polo compact hatchback car in Brazil",
            "a photo of a modern Volkswagen Polo subcompact two-box hatchback car on a road",
        ],
    },
    "Volkswagen Up": {
        "categoria": "Hatchback",
        "prompts": [
            "front view of a tiny Volkswagen Up micro city car with short snub nose hood and rounded headlights",
            "front three-quarters view of an ultra-compact Volkswagen Up micro city car with short front overhang",
            "rear view of a tiny Volkswagen Up subcompact city car with vertical corner taillights and flat vertical rear hatch",
            "rear angle of a tiny Volkswagen Up micro city car showing flat vertical tailgate without roof rails",
            "a photo of an ultra-compact Volkswagen Up micro city car in Brazil",
        ],
    },
    "Volkswagen Golf": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Volkswagen Golf medium hatchback car with wide horizontal grille, sleek headlights and iconic VW emblem",
            "front angle of a Volkswagen Golf hatchback showing low wide stance, sculpted hood and aggressive lower air dam",
            "rear view of a Volkswagen Golf medium hatchback car with horizontal split taillights and signature wide C-pillar",
            "rear angle of a Volkswagen Golf hatchback showing dual exhaust tips and aerodynamic rear spoiler",
            "a photo of a Volkswagen Golf five-door medium hatchback car on a road",
        ],
    },
    "Chevrolet Onix": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Chevrolet Onix compact hatchback car showing dual-port front grille, headlights and Chevrolet bowtie",
            "front angle of a modern Chevrolet Onix hatchback showing sculpted hood lines and lower air intake",
            "rear view of a Chevrolet Onix compact hatchback car with wide horizontal wraparound taillights",
            "rear angle of a modern Chevrolet Onix hatchback car in Brazil with aerodynamic curved tailgate",
            "a photo of a Chevrolet Onix subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Chevrolet Celta": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a small Chevrolet Celta economy hatchback car showing simple headlights and small front grille with gold bowtie",
            "front angle of a low Chevrolet Celta subcompact hatchback passenger car on a Brazilian street",
            "rear view of a Chevrolet Celta small economy hatchback with high-mounted triangular corner taillights",
            "rear angle of a Chevrolet Celta compact hatchback showing curved rear window and gold Chevrolet emblem",
            "a photo of a small low Chevrolet Celta two-box economy hatchback car in Brazil",
        ],
    },
    "Chevrolet Corsa": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Chevrolet Corsa compact hatchback car showing almond headlights and front grille with Chevrolet badge",
            "front angle of a classic Chevrolet Corsa subcompact hatchback car on a Brazilian street",
            "rear view of a Chevrolet Corsa hatchback with vertical taillights on the rear pillars",
            "rear angle of a compact Chevrolet Corsa hatchback passenger car",
            "a photo of a classic Chevrolet Corsa two-box compact hatchback car in Brazil",
        ],
    },
    "Chevrolet Cruze Sport6": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Chevrolet Cruze Sport6 medium hatchback showing aggressive dual-port grille, headlights and gold bowtie",
            "front angle of a Chevrolet Cruze Sport6 hatchback with sporty front bumper and sculpted hood",
            "rear view of a Chevrolet Cruze Sport6 hatchback car with prominent roof spoiler and wide horizontal taillights",
            "rear angle of a Chevrolet Cruze Sport6 medium hatchback showing aerodynamic fastback roofline",
            "a photo of a modern Chevrolet Cruze Sport6 five-door hatchback car on a Brazilian highway",
        ],
    },
    "Fiat Palio": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Fiat Palio compact hatchback car showing rounded headlights, front grille and Fiat emblem",
            "front angle of a Fiat Palio subcompact hatchback car on a Brazilian street",
            "rear view of a Fiat Palio compact hatchback with curved taillights mounted on the rear pillars",
            "rear angle of a Fiat Palio hatchback showing rounded tailgate and Fiat badge",
            "a photo of a Fiat Palio compact two-box hatchback car in Brazil",
        ],
    },
    "Fiat Uno": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Fiat Uno Vivace compact hatchback showing rounded-square front grille slots and large headlights",
            "front angle of a boxy Fiat Uno economy hatchback car on a Brazilian street",
            "rear view of a Fiat Uno compact hatchback car with tall vertical taillights flanking the flat tailgate",
            "rear angle of a Fiat Uno hatchback showing boxy silhouette and short rear bumper",
            "a photo of a Fiat Uno compact two-box hatchback car on a Brazilian road",
        ],
    },
    "Fiat Argo": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a modern Fiat Argo compact hatchback car showing wide textured grille, sharp headlights and Fiat lettering",
            "front angle of a Fiat Argo hatchback showing sculpted hood and sporty front bumper intake",
            "rear view of a Fiat Argo compact hatchback car with sharp horizontal split LED taillights",
            "rear angle of a modern Fiat Argo hatchback showing integrated tailgate spoiler and rear bumper diffuser",
            "a photo of a modern Fiat Argo subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Fiat Mobi": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a tiny Fiat Mobi subcompact city car showing high front hood, oversized headlights and wide Fiat grille",
            "front angle of a compact Fiat Mobi hatchback on a Brazilian street",
            "rear view of a Fiat Mobi subcompact car featuring distinctive glossy all-black glass rear hatch and corner taillights",
            "rear angle of a tiny Fiat Mobi city car showing vertical flat glass tailgate and compact bumper",
            "a photo of a small Fiat Mobi subcompact urban hatchback car in Brazil",
        ],
    },
    "Fiat Punto": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Fiat Punto hatchback car showing aerodynamic sloping front nose, narrow grille and headlights",
            "front angle of a Fiat Punto compact hatchback car with curved hood and sporty front fascia",
            "rear view of a Fiat Punto hatchback car with distinctive tall vertical taillights along the rear pillars",
            "rear angle of a Fiat Punto hatchback showing curved muscular rear shoulders and Fiat badge",
            "a photo of a Fiat Grande Punto compact hatchback car on a Brazilian road",
        ],
    },
    "Hyundai HB20": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Hyundai HB20 compact hatchback car showing large cascading front grille, sharp headlights and Hyundai logo",
            "front angle of a modern Hyundai HB20 hatchback showing sculpted hood creases and geometric fog light surrounds",
            "rear view of a Hyundai HB20 compact hatchback car with sharp angular boomerang taillights or full-width light bar",
            "rear angle of a modern Hyundai HB20 hatchback car in Brazil showing sculpted tailgate and rear bumper",
            "a photo of a Hyundai HB20 subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Ford Ka": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Ford Ka compact hatchback car showing trapezoidal honeycomb grille, swept headlights and Ford blue oval",
            "front angle of a modern Ford Ka hatchback showing rounded hood and lower front bumper air intake",
            "rear view of a Ford Ka compact hatchback car with triangular wraparound corner taillights",
            "rear angle of a Ford Ka hatchback showing clean tailgate with centered Ford emblem",
            "a photo of a Ford Ka subcompact two-box hatchback passenger car on a Brazilian road",
        ],
    },
    "Ford Fiesta": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Ford Fiesta compact hatchback car showing prominent trapezoidal chrome grille, sharp headlights and Ford badge",
            "front angle of a Ford Fiesta hatchback showing sculpted hood lines and aerodynamic front bumper",
            "rear view of a Ford Fiesta compact hatchback car with vertical pillar taillights or horizontal wraparound rear lamps",
            "rear angle of a sporty Ford Fiesta hatchback showing rear roof spoiler and curved tailgate",
            "a photo of a Ford Fiesta compact five-door hatchback car on a Brazilian road",
        ],
    },
    "Renault Kwid": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Renault Kwid compact mini crossover hatchback showing high front hood, black plastic wheel arches and large Renault chrome diamond",
            "front angle of a Renault Kwid subcompact hatchback showing split daytime running lights and rugged front bumper",
            "rear view of a small Renault Kwid compact car with rounded rectangular taillights and tall rear bumper",
            "rear angle of a Renault Kwid mini crossover hatchback showing elevated ride height and compact tailgate",
            "a photo of an entry-level Renault Kwid subcompact crossover hatchback car on a Brazilian street",
        ],
    },
    "Renault Sandero": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Renault Sandero compact hatchback car showing wide front grille, headlights and large Renault chrome diamond",
            "front angle of a Renault Sandero hatchback showing wide stance, contoured hood and lower fog lamps",
            "rear view of a Renault Sandero compact hatchback car with horizontal taillights extending into the tailgate",
            "rear angle of a spacious Renault Sandero hatchback car in Brazil showing wide tailgate and Renault emblem",
            "a photo of a Renault Sandero subcompact two-box hatchback passenger car on a Brazilian road",
        ],
    },
    "Peugeot 208": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Peugeot 208 compact hatchback car showing dramatic saber-tooth LED daytime running lights, wide front grille and Peugeot lion",
            "front angle of a modern Peugeot 208 hatchback showing sculpted hood and aggressive front bumper intake",
            "rear view of a Peugeot 208 hatchback car with glossy black horizontal band connecting three-claw LED taillights",
            "rear angle of a modern Peugeot 208 compact hatchback showing sporty rear spoiler and curved tailgate",
            "a photo of a modern Peugeot 208 subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "Citroën C3": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Citroen C3 compact hatchback car showing split headlights, double chevron chrome grille and high rounded hood",
            "front angle of a Citroen C3 hatchback showing elevated stance and black lower bumper claddings",
            "rear view of a Citroen C3 compact hatchback car with rectangular C-shaped taillights and upright tailgate",
            "rear angle of a modern Citroen C3 hatchback showing rounded rear bumper and double chevron badge",
            "a photo of a Citroen C3 compact five-door hatchback car on a Brazilian road",
        ],
    },
    "Honda Fit": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Honda Fit Jazz compact tall hatchback showing steeply sloped windshield, short hood and Honda H grille",
            "front angle of a Honda Fit MPV-style hatchback showing swept-back headlights and aerodynamic front bumper",
            "rear view of a Honda Fit Jazz hatchback car with tall vertical taillights flanking the rear glass window",
            "rear angle of a Honda Fit tall compact hatchback showing large vertical tailgate and chrome rear trim",
            "a photo of a Honda Fit subcompact one-box hatchback car on a Brazilian road",
        ],
    },
    "Toyota Etios": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Toyota Etios compact hatchback car showing smiling front grille, headlights and Toyota emblem",
            "front angle of a Toyota Etios subcompact hatchback car on a Brazilian street",
            "rear view of a Toyota Etios compact hatchback car with simple vertical corner taillights and flat tailgate",
            "rear angle of a Toyota Etios hatchback showing compact rear bumper and centered Toyota badge",
            "a photo of a Toyota Etios subcompact two-box hatchback passenger car in Brazil",
        ],
    },
    "Toyota Yaris": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a Toyota Yaris compact hatchback car showing large black lower grille, slim headlights and Toyota badge",
            "front angle of a Toyota Yaris hatchback showing sculpted front bumper, hood lines and integrated fog lights",
            "rear view of a Toyota Yaris compact hatchback car with horizontal wraparound split taillights",
            "rear angle of a modern Toyota Yaris hatchback showing sporty rear diffuser and tailgate spoiler",
            "a photo of a Toyota Yaris subcompact two-box hatchback car on a Brazilian road",
        ],
    },
    "BYD Dolphin": {
        "categoria": "Hatchback",
        "prompts": [
            "front three-quarters view of a BYD Dolphin electric compact hatchback car showing closed aerodynamic grille, modern LED headlights and BYD badge",
            "front angle of a BYD Dolphin EV hatchback showing short electric vehicle hood and two-tone exterior",
            "rear view of a BYD Dolphin EV hatchback car featuring full-width continuous LED light bar with geometric knot pattern",
            "rear angle of a modern BYD Dolphin electric hatchback showing futuristic rear spoiler and smooth bumper",
            "a photo of a BYD Dolphin all-electric compact hatchback car on a Brazilian road",
        ],
    },
    "Toyota Corolla": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Toyota Corolla medium sedan car showing wide lower trapezoidal grille, sharp LED headlights and Toyota emblem",
            "front angle of a modern Toyota Corolla sedan showing sculpted hood and aerodynamic three-box sedan silhouette",
            "rear view of a Toyota Corolla four-door sedan car with elongated separate rear trunk and horizontal LED taillights",
            "rear angle of a Toyota Corolla medium sedan showing separate boot lid, chrome accent bar and rear bumper",
            "a photo of a Toyota Corolla four-door three-box sedan passenger car on a Brazilian road",
        ],
    },
    "Honda Civic": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Honda Civic medium sedan car showing sporty solid wing front grille, jewel-eye headlights and Honda emblem",
            "front angle of a low aggressive Honda Civic sedan showing sculpted hood creases and wide stance",
            "rear view of a Honda Civic four-door sedan car with elongated separate trunk and distinctive boomerang C-shaped LED taillights",
            "rear angle of a sporty Honda Civic three-box sedan showing fastback trunk lid and dual exhaust cutouts",
            "a photo of a Honda Civic four-door passenger sedan car on a Brazilian road",
        ],
    },
    "Honda City": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Honda City compact sedan car showing thick chrome front grille bar, sharp headlights and Honda emblem",
            "front angle of a modern Honda City sedan showing aerodynamic front nose and sculpted three-box sedan proportions",
            "rear view of a Honda City four-door compact sedan car with elongated separate rear trunk and wraparound LED taillights",
            "rear angle of a Honda City sedan showing distinct rear boot lid, integrated lip spoiler and Honda badge",
            "a photo of a Honda City four-door subcompact three-box sedan car on a Brazilian road",
        ],
    },
    "Chevrolet Onix Plus": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Chevrolet Onix Plus compact sedan car showing dual-port grille, headlights and Chevrolet gold bowtie",
            "front angle of a modern Chevrolet Onix Plus sedan showing sleek hood lines and three-box sedan profile",
            "rear view of a Chevrolet Onix Plus four-door sedan car with elongated separate rear trunk and horizontal split taillights",
            "rear angle of a Chevrolet Onix Plus compact sedan showing separate rear trunk lid and Chevrolet badge",
            "a photo of a Chevrolet Onix Plus subcompact three-box sedan passenger car on a Brazilian road",
        ],
    },
    "Chevrolet Cruze": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Chevrolet Cruze medium sedan car showing dual-port aerodynamic grille, projector headlights and bowtie",
            "front angle of an elegant Chevrolet Cruze four-door sedan showing sculpted hood and chrome fog lamp bezels",
            "rear view of a Chevrolet Cruze medium sedan car with elongated separate rear trunk lid and horizontal taillights",
            "rear angle of a Chevrolet Cruze three-box sedan showing wide rear bumper, trunk lip and Chevrolet badge",
            "a photo of a Chevrolet Cruze four-door medium passenger sedan car on a Brazilian highway",
        ],
    },
    "Chevrolet Classic": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a small Chevrolet Classic compact sedan car showing simple front grille, trapezoidal headlights and gold bowtie",
            "front angle of a classic Chevrolet Corsa Classic four-door economy sedan on a Brazilian street",
            "rear view of a Chevrolet Classic small sedan car with tall separate rear trunk box and corner taillights",
            "rear angle of a Chevrolet Classic compact three-box sedan showing distinctive trunk lid with gold bowtie",
            "a photo of an economy Chevrolet Classic four-door three-box sedan car in Brazil",
        ],
    },
    "Volkswagen Voyage": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Volkswagen Voyage compact sedan car showing horizontal front grille, headlights and VW emblem",
            "front angle of a Volkswagen Voyage four-door subcompact sedan car on a Brazilian road",
            "rear view of a Volkswagen Voyage compact sedan car with elongated separate rear trunk and horizontal taillights",
            "rear angle of a Volkswagen Voyage three-box sedan showing distinct boot lid with centered VW logo",
            "a photo of a Volkswagen Voyage four-door subcompact three-box sedan car in Brazil",
        ],
    },
    "Volkswagen Virtus": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Volkswagen Virtus compact sedan car showing wide chrome grille, sharp angular headlights and VW logo",
            "front angle of a modern Volkswagen Virtus four-door sedan showing sculpted hood and lower bumper air intake",
            "rear view of a Volkswagen Virtus compact sedan car with elongated separate rear trunk and slim horizontal LED taillights",
            "rear angle of a Volkswagen Virtus three-box sedan showing elegant separate trunk lid and chrome rear trim",
            "a photo of a modern Volkswagen Virtus four-door three-box sedan car on a Brazilian road",
        ],
    },
    "Volkswagen Jetta": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Volkswagen Jetta medium executive sedan car showing wide hexagonal chrome grille, LED headlights and VW emblem",
            "front angle of a Volkswagen Jetta four-door sedan showing long hood creases and muscular front fascia",
            "rear view of a Volkswagen Jetta medium sedan car with elongated separate rear trunk lid and sharp horizontal LED taillights",
            "rear angle of an executive Volkswagen Jetta three-box sedan showing dual exhaust surrounds and wide trunk",
            "a photo of a Volkswagen Jetta four-door executive three-box sedan car on a Brazilian highway",
        ],
    },
    "Fiat Siena": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Fiat Grand Siena compact sedan car showing chrome front grille, swept-back headlights and Fiat badge",
            "front angle of a Fiat Siena four-door subcompact sedan car on a Brazilian street",
            "rear view of a Fiat Grand Siena compact sedan car with tall separate rear trunk and high-mounted horizontal taillights",
            "rear angle of a Fiat Siena three-box sedan showing elevated boot lid and centered Fiat emblem",
            "a photo of a Fiat Siena four-door subcompact three-box sedan passenger car in Brazil",
        ],
    },
    "Fiat Cronos": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Fiat Cronos compact sedan car showing stylish textured grille, sculpted hood, headlights and Fiat logo",
            "front angle of a modern Fiat Cronos four-door sedan showing aggressive lower air intake and LED daytime lights",
            "rear view of a Fiat Cronos compact sedan car with elongated separate rear trunk and slim horizontal LED taillights",
            "rear angle of a Fiat Cronos three-box sedan showing sculpted rear trunk lid with integrated ducktail spoiler",
            "a photo of a modern Fiat Cronos four-door three-box sedan car on a Brazilian road",
        ],
    },
    "Hyundai HB20S": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Hyundai HB20S compact sedan car showing large cascading front grille, sharp headlights and Hyundai logo",
            "front angle of a modern Hyundai HB20S four-door sedan showing sculpted hood creases and geometric bumper styling",
            "rear view of a Hyundai HB20S compact sedan car with elongated separate rear trunk and sharp angular taillights",
            "rear angle of a Hyundai HB20S three-box sedan showing fastback-style separate trunk lid and rear bumper",
            "a photo of a Hyundai HB20S four-door subcompact three-box sedan car on a Brazilian road",
        ],
    },
    "Renault Logan": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Renault Logan compact sedan car showing wide front grille, headlights and large Renault chrome diamond",
            "front angle of a Renault Logan four-door sedan showing tall front hood and upright three-box sedan profile",
            "rear view of a Renault Logan compact sedan car with tall boxy separate rear trunk and vertical rectangular taillights",
            "rear angle of a spacious Renault Logan three-box sedan showing wide flat trunk lid and Renault badge",
            "a photo of a Renault Logan four-door boxy compact sedan car on a Brazilian street",
        ],
    },
    "Nissan Versa": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Nissan Versa compact sedan car showing bold chrome V-motion front grille, sharp headlights and Nissan logo",
            "front angle of a modern Nissan Versa four-door sedan showing sculpted hood and lower front air dam",
            "rear view of a Nissan Versa compact sedan car with elongated separate rear trunk and wide boomerang split taillights",
            "rear angle of a Nissan Versa three-box sedan showing floating roof C-pillar and separate rear boot lid",
            "a photo of a modern Nissan Versa four-door three-box sedan car on a Brazilian road",
        ],
    },
    "Nissan Sentra": {
        "categoria": "Sedan",
        "prompts": [
            "front three-quarters view of a Nissan Sentra medium sedan car showing wide bold V-motion chrome grille, slim LED headlights and Nissan logo",
            "front angle of an aggressive Nissan Sentra four-door sedan showing low sculpted hood and wide sports sedan stance",
            "rear view of a Nissan Sentra medium sedan car with elongated separate rear trunk and sharp horizontal LED taillights",
            "rear angle of a Nissan Sentra executive three-box sedan showing sculpted trunk lid with integrated lip spoiler",
            "a photo of a modern Nissan Sentra four-door medium passenger sedan car on a Brazilian highway",
        ],
    },
    "Jeep Renegade": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Jeep Renegade compact boxy SUV with roof rack rails on top, tall ground clearance, iconic seven-slot grille and round headlights",
            "front angle of a Jeep Renegade compact crossover SUV showing high upright hood, black bumper cladding and round projector lamps",
            "rear view of a boxy Jeep Renegade compact SUV with roof rails and distinctive square taillights with embossed X pattern",
            "rear angle of a Jeep Renegade SUV showing tall upright tailgate, roof rack rails, black fender flares and high ground clearance",
            "a photo of a boxy Jeep Renegade compact crossover SUV car on a Brazilian road",
        ],
    },
    "Jeep Compass": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Jeep Compass medium SUV with roof rack rails on top, tall ground clearance, seven-slot front grille and slim LED headlights",
            "front angle of a modern Jeep Compass crossover SUV showing high sculpted hood, black wheel arch moldings and Jeep emblem",
            "rear view of a Jeep Compass medium SUV with roof rack rails on top and slim horizontal wraparound LED taillights",
            "rear angle of a Jeep Compass SUV showing dual-tone floating roof with rails, elevated suspension and wide tailgate",
            "a photo of a modern Jeep Compass medium crossover SUV car on a Brazilian road",
        ],
    },
    "Volkswagen T-Cross": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Volkswagen T-Cross compact crossover SUV with roof rack rails on top, wide horizontal grille and prominent VW emblem",
            "a photo of a Volkswagen T-Cross subcompact SUV crossover car in Brazil",
            "front angle of a modern Volkswagen T-Cross SUV showing tall front hood, roof rails and headlights",
            "a photo of a Volkswagen T-Cross SUV in Brazil with roof rack rails on top and wide horizontal grille",
            "a photo of a dark Volkswagen T-Cross compact crossover SUV car in Brazil",
            "rear view of a Volkswagen T-Cross compact SUV with black horizontal reflective light bar across the tailgate",
        ],
    },
    "Volkswagen Nivus": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Volkswagen Nivus coupe crossover SUV with roof rack rails on top, piano black grille, LED headlights and VW emblem",
            "front angle of a modern Volkswagen Nivus compact fastback crossover showing sculpted hood creases, tall ground clearance and fog lights",
            "rear view of a Volkswagen Nivus coupe SUV with sloping fastback rear window and gloss black horizontal band connecting taillights",
            "rear angle of a Volkswagen Nivus Taigo coupe crossover SUV showing sleek sloping roofline with roof rails and rear spoiler",
            "a photo of a modern Volkswagen Nivus compact coupe SUV crossover car on a Brazilian road",
        ],
    },
    "Hyundai Creta": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Hyundai Creta compact SUV with roof rack rails on top, tall ground clearance, large parametric front grille and split LED headlights",
            "front angle of a modern Hyundai Creta crossover SUV showing high commanding hood, black wheel arch cladding and Hyundai emblem",
            "rear view of a Hyundai Creta compact SUV with roof rack rails on top and distinctive geometric split LED taillights",
            "rear angle of a Hyundai Creta crossover SUV showing silver roof arch accent, roof rails and sculpted tailgate",
            "a photo of a modern Hyundai Creta compact crossover SUV car on a Brazilian road",
        ],
    },
    "Hyundai Tucson": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Hyundai Tucson medium SUV with roof rack rails on top, tall ground clearance, bold front grille and LED headlights",
            "front angle of a modern Hyundai Tucson crossover SUV showing sculpted muscular front fenders and high ground clearance",
            "rear view of a Hyundai Tucson medium SUV with roof rack rails on top and horizontal LED taillights",
            "rear angle of a Hyundai Tucson crossover SUV showing elevated suspension, roof rails and wide rear tailgate",
            "a photo of a Hyundai Tucson medium crossover SUV car on a Brazilian road",
        ],
    },
    "Chevrolet Tracker": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Chevrolet Tracker compact SUV with roof rack rails on top, tall ground clearance, dual-port grille and Chevrolet bowtie",
            "front angle of a modern Chevrolet Tracker crossover SUV showing high front hood, black lower bumper cladding and slim headlights",
            "rear view of a Chevrolet Tracker compact SUV with roof rack rails on top and high-mounted horizontal wraparound taillights",
            "rear angle of a Chevrolet Tracker crossover SUV showing elevated ground clearance, black wheel arches and roof rails",
            "a photo of a modern Chevrolet Tracker subcompact crossover SUV car on a Brazilian road",
        ],
    },
    "Nissan Kicks": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Nissan Kicks compact crossover SUV with roof rack rails on top, large V-motion front grille, sharp headlights and Nissan logo",
            "front angle of a modern Nissan Kicks SUV showing tall front hood, black lower bumper claddings and elevated ground clearance",
            "rear view of a Nissan Kicks compact SUV with roof rack rails on top and boomerang connected LED taillights",
            "rear angle of a Nissan Kicks crossover SUV showing floating roof design with roof rails and sporty tailgate spoiler",
            "a photo of a modern Nissan Kicks compact crossover SUV car on a Brazilian road",
        ],
    },
    "Honda HR-V": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Honda HR-V compact crossover SUV with roof rack rails on top, tall ground clearance, horizontal slat grille and slim LED headlights",
            "front angle of a modern Honda HR-V SUV showing sleek front hood, black wheel arch cladding and Honda emblem",
            "rear view of a Honda HR-V compact SUV with roof rack rails on top and full-width continuous horizontal LED taillight bar",
            "rear angle of a Honda HR-V coupe SUV showing hidden rear door handles in C-pillar, roof rails and sloping tailgate",
            "a photo of a modern Honda HR-V compact crossover SUV car on a Brazilian road",
        ],
    },
    "Renault Duster": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Renault Duster rugged compact SUV with heavy roof rack rails on top, tall ground clearance, muscular front bumper and Renault diamond",
            "front angle of a rugged Renault Duster SUV showing wide flared wheel arches, front skid plate and elevated stance",
            "rear view of a Renault Duster compact SUV with roof rack rails on top and distinctive square taillights with cross pattern",
            "rear angle of a Renault Duster rugged crossover SUV showing prominent roof rails, high ground clearance and wide tailgate",
            "a photo of a Renault Duster rugged compact SUV car on a Brazilian road",
        ],
    },
    "Ford EcoSport": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Ford EcoSport compact SUV with roof rack rails on top, tall ground clearance, large trapezoidal grille and Ford blue oval",
            "front angle of a Ford EcoSport mini SUV showing high hood, lower bumper cladding and projector headlights",
            "rear view of a Ford EcoSport compact SUV with distinctive external spare tire mounted on the rear swing tailgate and roof rack rails",
            "rear angle of a Ford EcoSport mini crossover SUV showing high ground clearance, roof rails and tailgate-mounted spare wheel",
            "a photo of a Ford EcoSport compact crossover SUV car on a Brazilian road",
        ],
    },
    "Fiat Pulse": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Fiat Pulse compact crossover SUV with roof rack rails on top, tall ground clearance, wide front grille and Fiat script logo",
            "front angle of a modern Fiat Pulse SUV showing sculpted high hood, black wheel arch cladding and LED headlights",
            "rear view of a Fiat Pulse compact crossover SUV with roof rack rails on top and horizontal tapered LED taillights",
            "rear angle of a Fiat Pulse compact SUV showing elevated suspension, roof rails and sporty rear bumper",
            "a photo of a modern Fiat Pulse compact crossover SUV car on a Brazilian road",
        ],
    },
    "Fiat Fastback": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Fiat Fastback coupe crossover SUV with roof rack rails on top, tall ground clearance, honeycomb grille and Fiat logo",
            "front angle of a Fiat Fastback SUV showing muscular front bumper, fog lamp intakes and full-LED headlights",
            "rear view of a Fiat Fastback coupe SUV with elevated fastback tail, integrated ducktail spoiler and slim horizontal LED taillights",
            "rear angle of a Fiat Fastback coupe crossover showing sweeping coupe roofline with roof rails and tall ground clearance",
            "a photo of a modern Fiat Fastback coupe crossover SUV car on a Brazilian road",
        ],
    },
    "Toyota Corolla Cross": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a Toyota Corolla Cross medium SUV with roof rack rails on top, tall ground clearance, double trapezoid front grille and Toyota emblem",
            "front angle of a Toyota Corolla Cross crossover SUV showing high sculpted hood, black fender cladding and sleek LED headlights",
            "rear view of a Toyota Corolla Cross medium SUV with roof rack rails on top and elongated horizontal LED taillights",
            "rear angle of a Toyota Corolla Cross SUV showing elevated suspension, roof rails on top and wide rear tailgate",
            "a photo of a modern Toyota Corolla Cross medium crossover SUV car on a Brazilian road",
        ],
    },
    "Toyota SW4": {
        "categoria": "SUV / Utilitário",
        "prompts": [
            "front three-quarters view of a large Toyota SW4 Fortuner SUV with roof rack rails on top, massive ground clearance, imposing chrome grille and LED headlights",
            "front angle of a large Toyota SW4 SUV showing high commanding front hood, fog lamp bezels and prominent Toyota emblem",
            "rear view of a large Toyota SW4 Fortuner SUV with roof rack rails on top, tall rear bumper and horizontal LED taillights with chrome SW4 bar",
            "rear angle of a heavy Toyota SW4 full-size SUV showing high ground clearance, side running boards, roof rails and rear spoiler",
            "a photo of a large Toyota SW4 Fortuner four-wheel drive SUV on a Brazilian road",
        ],
    },
    "Fiat Strada": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Fiat Strada compact pickup truck showing bold front grille with Fiat script logo, Italian flag badge and headlights",
            "front angle of a modern Fiat Strada small utility pickup truck on a Brazilian road",
            "rear view of a Fiat Strada compact pickup truck showing open rear cargo bed, vertical taillights and tailgate with Fiat lettering",
            "rear angle of a Fiat Strada double-cab pickup truck showing open cargo bed with bedliner, sports bar and utility stance",
            "a photo of a Fiat Strada compact utility pickup truck car in Brazil",
        ],
    },
    "Volkswagen Saveiro": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Volkswagen Saveiro compact utility pickup truck showing front grille, headlights and VW emblem",
            "front angle of a Volkswagen Saveiro small pickup truck with lower front bumper and fog lights",
            "rear view of a Volkswagen Saveiro compact pickup truck showing open rear cargo bed and tailgate with VW logo",
            "rear angle of a Volkswagen Saveiro utility pickup truck showing open cargo bed, sports bar roll cage and low side steps",
            "a photo of a Volkswagen Saveiro subcompact utility pickup truck on a Brazilian road",
        ],
    },
    "Fiat Toro": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Fiat Toro medium pickup truck showing split two-tier LED headlights, bold front grille and large Fiat lettering",
            "front angle of a modern Fiat Toro pickup truck showing muscular sculpted hood and rugged front skid plate",
            "rear view of a Fiat Toro medium pickup truck featuring unique dual-opening barn-door split tailgate and vertical taillights",
            "rear angle of a Fiat Toro pickup truck showing open rear cargo bed, sports bar roll bar, roof rails and utility stance",
            "a photo of a Fiat Toro compact unibody pickup truck on a Brazilian road",
        ],
    },
    "Toyota Hilux": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Toyota Hilux double-cab pickup truck showing massive trapezoidal front grille, headlights and Toyota emblem",
            "front angle of a rugged Toyota Hilux 4x4 pickup truck showing high front bumper, skid plate and commanding stance",
            "rear view of a Toyota Hilux pickup truck showing open rear cargo bed, vertical corner taillights and tailgate handle",
            "rear angle of a Toyota Hilux double-cab pickup truck showing large open cargo bed, chrome sports bar roll bar and side steps",
            "a photo of a rugged Toyota Hilux double-cab four-wheel drive pickup truck on a Brazilian road",
        ],
    },
    "Chevrolet S10": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Chevrolet S10 pickup truck showing wide horizontal chrome grille bar, headlights and Chevrolet bowtie",
            "front angle of a Chevrolet S10 double-cab pickup truck showing high commanding hood and lower bumper fog lamps",
            "rear view of a Chevrolet S10 pickup truck showing open rear cargo bed and tailgate with Chevrolet emblem",
            "rear angle of a Chevrolet S10 double-cab pickup truck showing open cargo bed with bedliner, sports bar roll bar and side steps",
            "a photo of a Chevrolet S10 midsize utility pickup truck on a Brazilian road",
        ],
    },
    "Ford Ranger": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Ford Ranger double-cab pickup truck showing bold front grille, signature C-clamp headlights and Ford emblem",
            "front angle of a modern Ford Ranger 4x4 pickup truck showing high muscular hood and rugged steel skid plate",
            "rear view of a Ford Ranger pickup truck showing open rear cargo bed and tailgate with stamped Ranger lettering",
            "rear angle of a Ford Ranger double-cab pickup truck showing open cargo bed with sports bar roll bar and elevated suspension",
            "a photo of a Ford Ranger double-cab utility pickup truck on a Brazilian road",
        ],
    },
    "Volkswagen Amarok": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Volkswagen Amarok double-cab pickup truck showing wide chrome horizontal grille, headlights and VW logo",
            "front angle of a Volkswagen Amarok 4x4 pickup truck showing muscular squared wheel arches and wide front bumper",
            "rear view of a Volkswagen Amarok pickup truck showing wide open rear cargo bed and tailgate with centered VW emblem",
            "rear angle of a Volkswagen Amarok double-cab pickup truck showing open cargo bed, chrome sports bar roll bar and 4Motion badge",
            "a photo of a Volkswagen Amarok double-cab midsize pickup truck on a Brazilian road",
        ],
    },
    "Chevrolet Montana": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Chevrolet Montana compact pickup truck showing split front headlights, dark grille and Chevrolet bowtie",
            "front angle of a modern Chevrolet Montana pickup truck showing sculpted hood and lower bumper skid plate",
            "rear view of a Chevrolet Montana compact pickup truck showing open rear cargo bed and modern tailgate with black accent trim",
            "rear angle of a Chevrolet Montana double-cab pickup truck showing open cargo bed, bedliner, sports bar and roof rails",
            "a photo of a modern Chevrolet Montana compact utility pickup truck on a Brazilian road",
        ],
    },
    "Renault Oroch": {
        "categoria": "Picape",
        "prompts": [
            "front three-quarters view of a Renault Oroch double-cab pickup truck showing rugged front bumper, headlights and Renault chrome diamond",
            "front angle of a Renault Duster Oroch pickup truck showing flared muscular fenders and high ground clearance",
            "rear view of a Renault Oroch pickup truck showing open rear cargo bed, tailgate with Oroch lettering and vertical taillights",
            "rear angle of a Renault Oroch double-cab pickup truck showing open cargo bed with black tubular sports bar roll bar",
            "a photo of a Renault Oroch compact double-cab utility pickup truck on a Brazilian road",
        ],
    },
    "Chevrolet Spin": {
        "categoria": "Minivan",
        "prompts": [
            "front three-quarters view of a Chevrolet Spin multi-purpose family minivan MPV showing tall front hood, dual-port grille, headlights and Chevrolet bowtie",
            "front angle of a Chevrolet Spin minivan car on a Brazilian street showing tall roofline and large windshield",
            "rear view of a Chevrolet Spin large family minivan MPV with high roofline, wide vertical rear hatch and horizontal taillights",
            "rear angle of a Chevrolet Spin 7-seater family minivan MPV showing tall boxy passenger cabin and roof rack rails on top",
            "a photo of a Chevrolet Spin family multi-purpose minivan MPV car in Brazil",
        ],
    },
    "Fiat Fiorino": {
        "categoria": "Furgão / Van",
        "prompts": [
            "front three-quarters view of a white Fiat Fiorino small cargo delivery box van showing compact front cab, headlights and Fiat badge",
            "front angle of a Fiat Fiorino commercial delivery panel van on a Brazilian street",
            "rear view of a white Fiat Fiorino utility cargo van showing tall cubic enclosed cargo box and asymmetric vertical split rear cargo doors",
            "rear angle of a Fiat Fiorino commercial panel van showing tall white delivery cargo box with no rear side windows",
            "a photo of a white Fiat Fiorino small commercial delivery cargo box van in Brazil",
        ],
    },
}

# Prompts Zero-Shot para categorias de carroceria (usados para validação cruzada ou fallback)
PROMPTS_CATEGORIAS = {
    "Hatchback": [
        "a compact hatchback car with a short rear overhang and rear hatch door",
        "rear view of a small two-box hatchback passenger car",
        "front view of a compact two-box hatchback passenger car",
    ],
    "Sedan": [
        "a three-box sedan passenger car with an elongated rear trunk lid",
        "rear view of a classic four-door sedan car with a separate trunk",
        "front view of a four-door sedan car with front hood and grille",
    ],
    "SUV / Utilitário": [
        "a sport utility vehicle SUV crossover with high ground clearance and large wheels",
        "rear view of a tall SUV crossover vehicle",
        "front view of a tall SUV crossover vehicle with high ground clearance",
    ],
    "Picape": [
        "a pickup truck with an open cargo bed in the back",
        "rear view of a pickup truck tailgate and cargo bed",
        "front view of a rugged pickup truck showing grille and headlights",
    ],
    "Minivan": [
        "a long multi-purpose family minivan MPV passenger van",
        "front view of a family minivan MPV passenger car",
    ],
    "Furgão / Van": [
        "a commercial cargo delivery van utility panel van",
        "front view of a commercial delivery panel van",
    ],
}

# Mapeamento auxiliar por substring (caso use modelo YOLO supervisionado)
MAPA_MODELOS_CATEGORIA = {
    k.lower().split()[-1]: v["categoria"] for k, v in CATALOGO_FROTA_BR.items()
}
MAPA_MODELOS_CATEGORIA.update({
    "fox": "Hatchback", "gol": "Hatchback", "palio": "Hatchback", "uno": "Hatchback",
    "onix": "Hatchback", "hb20": "Hatchback", "polo": "Hatchback", "celta": "Hatchback",
    "corolla": "Sedan", "civic": "Sedan", "voyage": "Sedan", "siena": "Sedan", "virtus": "Sedan",
    "compass": "SUV / Utilitário", "renegade": "SUV / Utilitário", "creta": "SUV / Utilitário",
    "t-cross": "SUV / Utilitário", "tracker": "SUV / Utilitário", "kicks": "SUV / Utilitário",
    "strada": "Picape", "toro": "Picape", "saveiro": "Picape", "hilux": "Picape", "s10": "Picape",
    "spin": "Minivan", "fiorino": "Furgão / Van",
})


def inferir_categoria_por_modelo(nome_modelo: str) -> str:
    """Inferir a carroceria/categoria a partir do nome ou modelo do veículo."""
    if nome_modelo in CATALOGO_FROTA_BR:
        return CATALOGO_FROTA_BR[nome_modelo]["categoria"]
    nome_norm = nome_modelo.lower()
    for chave, cat in MAPA_MODELOS_CATEGORIA.items():
        if chave in nome_norm:
            return cat
    return "Carro"


def formatar_nome_veiculo(classe: str) -> str:
    """
    Formata strings como 'vw_fox', 'chevrolet-onix', 'volkswagen_fox_2013'
    para 'Volkswagen Fox', 'Chevrolet Onix', etc.
    """
    limpo = classe.replace("-", " ").replace("_", " ")
    substituicoes = {
        "vw": "Volkswagen",
        "gm": "Chevrolet",
        "chevrolet": "Chevrolet",
        "volkswagen": "Volkswagen",
        "fiat": "Fiat",
        "ford": "Ford",
        "hyundai": "Hyundai",
        "toyota": "Toyota",
        "honda": "Honda",
        "renault": "Renault",
        "nissan": "Nissan",
        "jeep": "Jeep",
        "peugeot": "Peugeot",
        "citroen": "Citroën",
        "citroën": "Citroën",
        "mitsubishi": "Mitsubishi",
    }
    partes = limpo.split()
    partes_fmt = []
    for p in partes:
        p_low = p.lower()
        if p_low in substituicoes:
            partes_fmt.append(substituicoes[p_low])
        elif p.isdigit() and len(p) == 4:
            continue
        else:
            partes_fmt.append(p.capitalize())
    return " ".join(partes_fmt).strip() or classe


def estimar_cor_lataria(crop_veiculo: np.ndarray) -> str:
    """
    Analisa a região representativa da lataria do veículo no espaço de cores HSV
    e retorna a cor predominante em português.
    """
    if crop_veiculo is None or crop_veiculo.size == 0:
        return "Indefinida"

    h, w = crop_veiculo.shape[:2]
    if h < 20 or w < 20:
        return "Indefinida"

    # Seleciona a faixa central da lataria (desconsidera teto/vidro e rodas/asfalto)
    y1, y2 = int(h * 0.30), int(h * 0.75)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    lataria = crop_veiculo[y1:y2, x1:x2]

    if lataria.size == 0:
        lataria = crop_veiculo

    hsv = cv2.cvtColor(lataria, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Filtra reflexos brancos extremos de farol (> 240) e sombras pretas puras (< 30)
    mascara = (V > 30) & (V < 240)
    if np.sum(mascara) < 50:
        mascara = np.ones(V.shape, dtype=bool)

    s_media = float(np.mean(S[mascara]))
    v_media = float(np.mean(V[mascara]))
    h_media = float(np.mean(H[mascara]))

    # Cores acromáticas (baixa saturação: Branco, Preto, Prata / Cinza)
    if s_media < 65:
        if v_media < 65:
            return "Preto"
        elif v_media > 145:
            return "Branco"
        else:
            return "Prata / Cinza"

    # Cores cromáticas baseadas no Matiz (Hue no OpenCV vai de 0 a 179)
    if h_media < 12 or h_media >= 165:
        return "Vermelho"
    elif 12 <= h_media < 35:
        return "Amarelo / Laranja"
    elif 35 <= h_media < 85:
        return "Verde"
    elif 85 <= h_media < 135:
        return "Azul"
    elif 135 <= h_media < 165:
        return "Roxo / Vinho"

    return "Outra"


def classificar_tipo_veiculo(cls_id: int) -> str:
    """Retorna o tipo de veículo em português a partir do ID do COCO."""
    return MAPA_TIPOS_VEICULO.get(cls_id, "Carro")


def calcular_matriz_confiabilidade(prob_1: float, prob_2: float = 0.0) -> str:
    """
    Calcula a Matriz de Confiabilidade da predição VMMR baseado na probabilidade
    do 1º colocado e na margem de separação em relação ao 2º colocado:
    - Alta: prob_1 >= 0.40 e margem (prob_1 - prob_2) >= 0.25
    - Média: margem (prob_1 - prob_2) >= 0.10
    - Baixa / Ambígua: margem estreita (< 0.10) ou baixa certeza
    """
    conf_alta_min = getattr(config, "VMMR_CONF_ALTA_MIN", 0.40)
    margem_alta = getattr(config, "VMMR_MARGEM_ALTA", 0.25)
    margem_media = getattr(config, "VMMR_MARGEM_MEDIA", 0.10)

    margem = prob_1 - prob_2
    if prob_1 >= conf_alta_min and margem >= margem_alta:
        return "Alta"
    elif margem >= margem_media:
        return "Média"
    else:
        return "Baixa / Ambígua"


def _extrair_tensor_embedding(saida_modelo) -> torch.Tensor:
    """Garante compatibilidade entre diferentes versões do HuggingFace Transformers."""
    if isinstance(saida_modelo, torch.Tensor):
        return saida_modelo
    if hasattr(saida_modelo, "pooler_output") and saida_modelo.pooler_output is not None:
        return saida_modelo.pooler_output
    if hasattr(saida_modelo, "last_hidden_state"):
        return saida_modelo.last_hidden_state[:, 0]
    return saida_modelo[0]


class ClassificadorMarcaModelo:
    """
    Classificador de Marca, Modelo e Carroceria (Fase 2 - VMMR).
    Por padrão utiliza o motor Zero-Shot Google SigLIP ('google/siglip-base-patch16-224')
    com Prompt Ensembling da frota brasileira e cache dos vetores de texto em RAM/disco.
    """

    def __init__(self):
        self.habilitado = getattr(config, "CAR_CLASSIFIER_ENABLED", True)
        self.engine = getattr(config, "CAR_CLASSIFIER_ENGINE", "siglip")
        self.device = "cuda" if (config.USE_GPU and torch.cuda.is_available()) else "cpu"

        self.siglip_model = None
        self.siglip_processor = None
        self.nomes_modelos = []
        self.vetores_modelos = None     # Shape: [N_modelos, D]
        self.nomes_categorias = []
        self.vetores_categorias = None  # Shape: [N_categorias, D]

        self.modelo_yolo_cls = None

        if not self.habilitado:
            return

        # Opção A: Motor Zero-Shot Google SigLIP (Padrão)
        if self.engine == "siglip":
            if self._inicializar_siglip():
                return

        # Opção B: Modelo supervisionado YOLOv8-cls (.pt) caso configurado
        caminho_pt = getattr(
            config,
            "CAR_MODEL_WEIGHTS_PATH",
            os.path.join(config.BASE_DIR, "models", "car_yolov8_cls.pt"),
        )
        if os.path.exists(caminho_pt):
            try:
                from ultralytics import YOLO
                self.modelo_yolo_cls = YOLO(caminho_pt)
                print(f"[ClassificadorMarcaModelo] Modelo YOLO-cls carregado: {caminho_pt}")
            except Exception as e:
                print(f"[ClassificadorMarcaModelo] Aviso ao carregar YOLO-cls ({caminho_pt}): {e}")

    def _calcular_hash_catalogo(self, modelo_id: str) -> str:
        """Gera um hash único do catálogo de prompts + modelo para validar o cache em disco."""
        conteudo = f"{modelo_id}|{repr(CATALOGO_FROTA_BR)}|{repr(PROMPTS_CATEGORIAS)}"
        return hashlib.sha256(conteudo.encode("utf-8")).hexdigest()[:16]

    def _inicializar_siglip(self) -> bool:
        """Carrega o modelo SigLIP e pré-calcula (ou restaura do cache) os vetores de texto."""
        modelo_id = getattr(config, "SIGLIP_MODEL_ID", "google/siglip-base-patch16-224")
        caminho_cache = getattr(
            config,
            "SIGLIP_CACHE_PATH",
            os.path.join(config.BASE_DIR, "models", "siglip_text_embeddings.pt"),
        )
        try:
            from transformers import AutoProcessor, AutoModel

            print(f"[ClassificadorMarcaModelo] Carregando SigLIP '{modelo_id}' em {self.device}...")
            try:
                self.siglip_processor = AutoProcessor.from_pretrained(modelo_id, local_files_only=True)
                self.siglip_model = AutoModel.from_pretrained(modelo_id, local_files_only=True).to(self.device).eval()
            except Exception:
                self.siglip_processor = AutoProcessor.from_pretrained(modelo_id)
                self.siglip_model = AutoModel.from_pretrained(modelo_id).to(self.device).eval()

            hash_atual = self._calcular_hash_catalogo(modelo_id)

            # 1. Tenta carregar os vetores de texto já pré-calculados do disco (instantâneo)
            if os.path.exists(caminho_cache):
                try:
                    dados_cache = torch.load(caminho_cache, map_location=self.device, weights_only=True)
                    if dados_cache.get("hash") == hash_atual:
                        self.nomes_modelos = dados_cache["nomes_modelos"]
                        self.vetores_modelos = dados_cache["vetores_modelos"].to(self.device)
                        self.nomes_categorias = dados_cache["nomes_categorias"]
                        self.vetores_categorias = dados_cache["vetores_categorias"].to(self.device)
                        print(
                            f"[ClassificadorMarcaModelo] Cache de texto SigLIP carregado "
                            f"({len(self.nomes_modelos)} modelos BR prontos na RAM)."
                        )
                        return True
                except Exception:
                    pass

            # 2. Pré-calcula os vetores de texto com Prompt Ensembling UMA ÚNICA VEZ
            print(f"[ClassificadorMarcaModelo] Pré-calculando embeddings de texto ({len(CATALOGO_FROTA_BR)} modelos BR)...")
            nomes_mod = []
            lista_vetores_mod = []

            with torch.no_grad():
                for nome_carro, info in CATALOGO_FROTA_BR.items():
                    prompts = info["prompts"]
                    inputs_txt = self.siglip_processor(
                        text=prompts, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(self.device)
                    emb = _extrair_tensor_embedding(self.siglip_model.get_text_features(**inputs_txt))
                    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                    # Média dos prompts do mesmo modelo (Prompt Ensembling) e re-normalização
                    emb_medio = emb.mean(dim=0, keepdim=True)
                    emb_medio = emb_medio / emb_medio.norm(p=2, dim=-1, keepdim=True)
                    nomes_mod.append(nome_carro)
                    lista_vetores_mod.append(emb_medio)

                self.nomes_modelos = nomes_mod
                self.vetores_modelos = torch.cat(lista_vetores_mod, dim=0)

                nomes_cat = []
                lista_vetores_cat = []
                for nome_cat, prompts_cat in PROMPTS_CATEGORIAS.items():
                    inputs_cat = self.siglip_processor(
                        text=prompts_cat, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(self.device)
                    emb_c = _extrair_tensor_embedding(self.siglip_model.get_text_features(**inputs_cat))
                    emb_c = emb_c / emb_c.norm(p=2, dim=-1, keepdim=True)
                    emb_c_medio = emb_c.mean(dim=0, keepdim=True)
                    emb_c_medio = emb_c_medio / emb_c_medio.norm(p=2, dim=-1, keepdim=True)
                    nomes_cat.append(nome_cat)
                    lista_vetores_cat.append(emb_c_medio)

                self.nomes_categorias = nomes_cat
                self.vetores_categorias = torch.cat(lista_vetores_cat, dim=0)

            # Salva no disco para inicializações futuras instantâneas
            try:
                os.makedirs(os.path.dirname(caminho_cache), exist_ok=True)
                torch.save(
                    {
                        "hash": hash_atual,
                        "nomes_modelos": self.nomes_modelos,
                        "vetores_modelos": self.vetores_modelos.cpu(),
                        "nomes_categorias": self.nomes_categorias,
                        "vetores_categorias": self.vetores_categorias.cpu(),
                    },
                    caminho_cache,
                )
            except Exception:
                pass

            print(
                f"[ClassificadorMarcaModelo] SigLIP pronto! "
                f"{len(self.nomes_modelos)} modelos da frota BR cacheados em RAM."
            )
            return True

        except Exception as e:
            print(f"[ClassificadorMarcaModelo] Aviso ao inicializar SigLIP: {e}")
            return False

    def analisar_top_k(self, crop_veiculo: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """
        Retorna os Top-K modelos mais prováveis [(nome_modelo, probabilidade), ...]
        para inspeção e diagnóstico em cima de um recorte BGR (OpenCV) ou RGB.
        """
        if self.siglip_model is None or self.vetores_modelos is None:
            return []

        img_rgb = cv2.cvtColor(crop_veiculo, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        with torch.no_grad():
            inputs_img = self.siglip_processor(images=pil_img, return_tensors="pt").to(self.device)
            vetor_img = _extrair_tensor_embedding(self.siglip_model.get_image_features(**inputs_img))
            vetor_img = vetor_img / vetor_img.norm(p=2, dim=-1, keepdim=True)

            escala = self.siglip_model.logit_scale.exp()
            logits = vetor_img @ self.vetores_modelos.T * escala
            if hasattr(self.siglip_model, "logit_bias"):
                logits = logits + self.siglip_model.logit_bias
            probs = logits.softmax(dim=-1)[0]

            k_efetivo = min(k, len(self.nomes_modelos))
            top_probs, top_indices = torch.topk(probs, k_efetivo)

        return [
            (self.nomes_modelos[int(idx)], float(prob.item()))
            for prob, idx in zip(top_probs, top_indices)
        ]

    def calcular_confiabilidade(self, prob_1: float, prob_2: float = 0.0) -> str:
        """Calcula o nível na Matriz de Confiabilidade VMMR para o resultado."""
        return calcular_matriz_confiabilidade(prob_1, prob_2)

    def classificar(self, crop_veiculo: np.ndarray, tipo_veiculo: str = "Carro") -> dict:
        """
        Classifica a marca, modelo e categoria do veículo a partir de seu recorte (crop BGR).
        Retorna dicionário com:
            - marca_modelo: modelo vencedor
            - categoria: carroceria
            - confianca: score do 1º
            - top3: lista com os 3 mais prováveis e suas porcentagens formatadas
                    (ex: "1º Modelo (X%), 2º Modelo (Y%), 3º Modelo (Z%)")
            - confiabilidade: "Alta", "Média" ou "Baixa / Ambígua"
        """
        if not self.habilitado or crop_veiculo is None or crop_veiculo.size == 0:
            return {
                "marca_modelo": "-",
                "categoria": tipo_veiculo,
                "confianca": 0.0,
                "top3": "-",
                "confiabilidade": "Baixa / Ambígua",
            }

        h, w = crop_veiculo.shape[:2]
        if h < 40 or w < 40:
            return {
                "marca_modelo": "-",
                "categoria": tipo_veiculo,
                "confianca": 0.0,
                "top3": "-",
                "confiabilidade": "Baixa / Ambígua",
            }

        if tipo_veiculo != "Carro":
            return {
                "marca_modelo": tipo_veiculo,
                "categoria": tipo_veiculo,
                "confianca": 1.0,
                "top3": f"1º {tipo_veiculo} (100.0%)",
                "confiabilidade": "Alta",
            }

        # 1. Classificação Zero-Shot via Google SigLIP (Vision Encoder -> Produto Escalar com Cache RAM)
        if self.siglip_model is not None and self.vetores_modelos is not None:
            try:
                img_rgb = cv2.cvtColor(crop_veiculo, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                with torch.no_grad():
                    inputs_img = self.siglip_processor(images=pil_img, return_tensors="pt").to(self.device)
                    vetor_img = _extrair_tensor_embedding(self.siglip_model.get_image_features(**inputs_img))
                    vetor_img = vetor_img / vetor_img.norm(p=2, dim=-1, keepdim=True)

                    escala = self.siglip_model.logit_scale.exp()

                    # Similaridade contra os modelos específicos da frota BR
                    logits_mod = vetor_img @ self.vetores_modelos.T * escala
                    if hasattr(self.siglip_model, "logit_bias"):
                        logits_mod = logits_mod + self.siglip_model.logit_bias
                    probs_mod = logits_mod.softmax(dim=-1)[0]

                    k_efetivo = min(3, len(self.nomes_modelos))
                    top_probs, top_indices = torch.topk(probs_mod, k_efetivo)
                    top_probs_list = [float(p.item()) for p in top_probs]
                    top_nomes_list = [self.nomes_modelos[int(idx.item())] for idx in top_indices]

                    conf_mod = top_probs_list[0]
                    melhor_modelo = top_nomes_list[0]
                    prob_segundo = top_probs_list[1] if len(top_probs_list) > 1 else 0.0

                    # Formata Top-3 formatado: "1º Modelo (X%), 2º Modelo (Y%), 3º Modelo (Z%)"
                    top3_str = ", ".join(
                        f"{i+1}º {nome} ({p * 100:.1f}%)"
                        for i, (nome, p) in enumerate(zip(top_nomes_list, top_probs_list))
                    )

                    # Calcula Matriz de Confiabilidade
                    confiabilidade = self.calcular_confiabilidade(conf_mod, prob_segundo)

                    # Similaridade contra as categorias de carroceria (custo zero extra de imagem)
                    logits_cat = vetor_img @ self.vetores_categorias.T * escala
                    probs_cat = logits_cat.softmax(dim=-1)[0]
                    conf_cat_prob, top_cat_idx = torch.max(probs_cat, dim=0)
                    melhor_cat = self.nomes_categorias[int(top_cat_idx.item())]

                min_conf = getattr(config, "CAR_CLASSIFIER_MIN_CONF", 0.08)
                if conf_mod >= min_conf:
                    categoria_modelo = inferir_categoria_por_modelo(melhor_modelo)
                    return {
                        "marca_modelo": melhor_modelo,
                        "categoria": categoria_modelo,
                        "confianca": round(conf_mod, 3),
                        "top3": top3_str,
                        "confiabilidade": confiabilidade,
                    }
                else:
                    return {
                        "marca_modelo": melhor_cat,
                        "categoria": melhor_cat,
                        "confianca": round(float(conf_cat_prob.item()), 3),
                        "top3": top3_str,
                        "confiabilidade": "Baixa / Ambígua",
                    }
            except Exception as e:
                print(f"[ClassificadorMarcaModelo] Erro na inferência SigLIP: {e}")

        # 2. Alternativa: Inferência com YOLOv8-cls caso configurado
        if self.modelo_yolo_cls is not None:
            try:
                resultados = self.modelo_yolo_cls(crop_veiculo, verbose=False)
                if resultados and len(resultados) > 0:
                    res = resultados[0]
                    top1_idx = int(res.probs.top1)
                    nome_classe = str(res.names[top1_idx])
                    conf = float(res.probs.top1conf)

                    top5_indices = res.probs.top5 if hasattr(res.probs, "top5") else [top1_idx]
                    top_itens = []
                    for idx in top5_indices[:3]:
                        n = formatar_nome_veiculo(str(res.names[idx]))
                        p = float(res.probs.data[idx])
                        top_itens.append((n, p))
                    top3_str = ", ".join(f"{i+1}º {n} ({p*100:.1f}%)" for i, (n, p) in enumerate(top_itens))
                    prob_segundo = top_itens[1][1] if len(top_itens) > 1 else 0.0
                    confiabilidade = self.calcular_confiabilidade(conf, prob_segundo)

                    if conf >= getattr(config, "CAR_CLASSIFIER_MIN_CONF", 0.30):
                        marca_modelo_fmt = formatar_nome_veiculo(nome_classe)
                        categoria = inferir_categoria_por_modelo(marca_modelo_fmt)
                        return {
                            "marca_modelo": marca_modelo_fmt,
                            "categoria": categoria,
                            "confianca": round(conf, 2),
                            "top3": top3_str,
                            "confiabilidade": confiabilidade,
                        }
            except Exception:
                pass

        return {
            "marca_modelo": "Carro",
            "categoria": "Carro",
            "confianca": 0.0,
            "top3": "-",
            "confiabilidade": "Baixa / Ambígua",
        }


def classificar_veiculo(crop_veiculo: np.ndarray, cls_id: int = 2,
                        classificador_mm: ClassificadorMarcaModelo | None = None) -> dict:
    """
    Retorna dicionário consolidado com tipo, cor, e marca/modelo do veículo.
    """
    tipo = classificar_tipo_veiculo(cls_id)
    cor = estimar_cor_lataria(crop_veiculo)

    marca_modelo = "-"
    categoria = tipo
    confianca = 0.0
    top3 = "-"
    confiabilidade = "Baixa / Ambígua"

    if classificador_mm is not None:
        info_mm = classificador_mm.classificar(crop_veiculo, tipo)
        marca_modelo = info_mm.get("marca_modelo", "-")
        categoria = info_mm.get("categoria", tipo)
        confianca = info_mm.get("confianca", 0.0)
        top3 = info_mm.get("top3", "-")
        confiabilidade = info_mm.get("confiabilidade", "Baixa / Ambígua")

    return {
        "tipo": tipo,
        "cor": cor,
        "marca_modelo": marca_modelo,
        "categoria": categoria,
        "confianca": confianca,
        "top3": top3,
        "confiabilidade": confiabilidade,
    }
