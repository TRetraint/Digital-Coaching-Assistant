import numpy as np
import math

def Right_Up_Angle(coordinates):
    var1 = coordinates[8] - coordinates[6]
    var2 = coordinates[6] - coordinates[4]
    var12 = var1 * var2

    var3 = coordinates[9] - coordinates[7]
    var4 = coordinates[7] - coordinates[5]
    var34 = var3 * var4
    top = var12 + var34

    down = (math.sqrt(var1**2 + var3**2)) *(math.sqrt(var2**2 + var4**2))
    try:
        result = top/down
    except ZeroDivisionError:
        result = 0
    if result > 1:
        result = 1
    result = math.acos(result)
    result = (result*180)/math.pi
    return result

def Left_Up_Angle(coordinates):
    var1 = coordinates[14] - coordinates[12]
    var2 = coordinates[12] - coordinates[10]
    var12 = var1 * var2

    var3 = coordinates[15] - coordinates[13]
    var4 = coordinates[13] - coordinates[11]
    var34 = var3 * var4
    top = var12 + var34

    down = (math.sqrt(var1**2 + var3**2)) *(math.sqrt(var2**2 + var4**2))
    try:
        result = top/down
    except ZeroDivisionError:
        result = 0
    if result > 1:
        result = 1
    result = math.acos(result)
    result = (result*180)/math.pi
    return result

def Right_Low_Angle(coordinates):
    var1 = coordinates[20] - coordinates[18]
    var2 = coordinates[18] - coordinates[16]
    var12 = var1 * var2

    var3 = coordinates[21] - coordinates[19]
    var4 = coordinates[19] - coordinates[17]
    var34 = var3 * var4
    top = var12 + var34

    down = (math.sqrt(var1**2 + var3**2)) *(math.sqrt(var2**2 + var4**2))
    try:
        result = top/down
    except ZeroDivisionError:
        result = 0
    if result > 1:
        result = 1
    result = math.acos(result)
    result = (result*180)/math.pi
    return result

def Left_Low_Angle(coordinates):
    var1 = coordinates[26] - coordinates[24]
    var2 = coordinates[24] - coordinates[22]
    var12 = var1 * var2

    var3 = coordinates[27] - coordinates[25]
    var4 = coordinates[25] - coordinates[23]
    var34 = var3 * var4
    top = var12 + var34

    down = (math.sqrt(var1**2 + var3**2)) *(math.sqrt(var2**2 + var4**2))
    try:
        result = top/down
    except ZeroDivisionError:
        result = 0
    if result > 1:
        result = 1
    result = math.acos(result)
    result = (result*180)/math.pi
    return result


