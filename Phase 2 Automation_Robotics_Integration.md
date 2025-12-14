# Phase 2: Automation & Robotics Integration

## TL;DR - คำตอบสำหรับคำถามคุณ

| คำถาม | คำตอบ |
|-------|-------|
| เป็นไปได้ไหม? | **ได้ 100%** - Architecture ปัจจุบันรองรับอยู่แล้ว |
| ต้องแก้อะไรเยอะไหม? | **ไม่เยอะ** - เพิ่มแค่ 2-3 files ใน domain/infrastructure |
| ต้องรื้อระบบเดิมไหม? | **ไม่ต้อง** - AI core ใช้เหมือนเดิม 100% |

---

## ทำไมถึงต่อยอดได้ง่าย?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    สิ่งที่คุณมีตอนนี้ (Phase 1)                          │
│                                                                         │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐             │
│    │   Camera    │────▶│  AI Brain   │────▶│   Output    │             │
│    │  (ESP32)    │     │ (Classify)  │     │  (LED/จอ)   │             │
│    └─────────────┘     └─────────────┘     └─────────────┘             │
│                              │                                         │
│                              ▼                                         │
│                     "ชิ้นนี้คือ PET"                                    │
│                     "confidence: 0.95"                                 │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               │ ไม่ต้องแก้!
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    สิ่งที่จะเพิ่ม (Phase 2)                              │
│                                                                         │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐             │
│    │   Camera    │────▶│  AI Brain   │────▶│  Actuator   │             │
│    │ (Industrial)│     │ (เหมือนเดิม) │     │ (แขนกล/Gate)│             │
│    └─────────────┘     └─────────────┘     └─────────────┘             │
│          │                   │                    │                    │
│          ▼                   ▼                    ▼                    │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐             │
│    │  Conveyor   │     │    Queue    │     │   PLC/MCU   │             │
│    │   Belt      │     │ (Redis/MQ)  │     │  (Control)  │             │
│    └─────────────┘     └─────────────┘     └─────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

**หลักการสำคัญ:**
- **Decision System** (AI) แยกจาก **Execution System** (Hardware)
- คุณทำถูกตั้งแต่แรก!

---

## สิ่งที่ไม่ต้องแก้ (ใช้เหมือนเดิม 100%)

| Component | เหตุผล |
|-----------|--------|
| Domain Layer | Entities, Enums, Interfaces ยังใช้ได้ |
| Application Layer | ClassifyTrashUseCase, RoutingService ไม่เปลี่ยน |
| AI Models | Trash-Net, PlasticNet ทำงานเหมือนเดิม |
| IClassifier Interface | ไม่ต้องแก้ |
| ClassificationResult | เพิ่ม metadata ได้โดยไม่ breaking |
| Confidence Logic | L1/L2 threshold ใช้ต่อได้ |

---

## สิ่งที่ต้องเพิ่ม (น้อยมาก)

### 1. เพิ่ม Interface: IActuator (Domain Layer)

```python
# src/domain/interfaces/actuator.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from domain.enums.trash_category import TrashCategory, PlasticType

class IActuator(ABC):
    """Abstract interface สำหรับอุปกรณ์ทางกายภาพ"""

    @abstractmethod
    async def execute(self, command: 'ActuatorCommand') -> bool:
        """Execute physical action"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current actuator status"""
        pass

@dataclass
class ActuatorCommand:
    """Command สำหรับ Actuator"""
    target_bin: int                    # ช่องที่ต้องทิ้ง (1-6)
    category: TrashCategory            # ประเภทหลัก
    sub_category: PlasticType = None   # ประเภทย่อย (ถ้ามี)
    position_mm: float = 0             # ตำแหน่งบนสายพาน (mm)
    confidence: float = 0              # ความมั่นใจ
    timestamp: float = 0               # เวลาที่ตรวจจับ
```

**หมายเหตุ:** Domain layer รู้จัก "concept" ของ actuator แต่ไม่รู้ว่าเป็นแขนกลหรือ gate

---

### 2. เพิ่ม Use Case: DispatchAction (Application Layer)

```python
# src/application/dispatch_action.py
from dataclasses import dataclass
from typing import Optional
from domain.entities.classification_result import ClassificationResult
from domain.interfaces.actuator import IActuator, ActuatorCommand
from domain.enums.trash_category import TrashCategory

# Mapping: Category → Bin Number
BIN_MAPPING = {
    TrashCategory.CARDBOARD: 1,
    TrashCategory.GLASS: 2,
    TrashCategory.METAL: 3,
    TrashCategory.PAPER: 4,
    TrashCategory.PLASTIC: 5,
    TrashCategory.TRASH: 6,  # Reject bin
}

class DispatchActionUseCase:
    """Use case: สั่งงาน Actuator ตามผล Classification"""

    def __init__(self, actuator: IActuator):
        self._actuator = actuator

    async def execute(
        self,
        result: ClassificationResult,
        position_mm: float = 0
    ) -> bool:
        """
        สั่ง actuator ทำงานตามผล classification

        Args:
            result: ผลจาก ClassifyTrashUseCase
            position_mm: ตำแหน่งของขยะบนสายพาน
        """
        # ใช้ sub_category ถ้ามี (L2 result)
        category = result.category
        sub_category = None

        if result.sub_result:
            sub_category = result.sub_result.category

        # สร้าง command
        command = ActuatorCommand(
            target_bin=BIN_MAPPING.get(category, 6),  # Default: reject
            category=category,
            sub_category=sub_category,
            position_mm=position_mm,
            confidence=result.confidence,
            timestamp=result.metadata.get("timestamp", 0)
        )

        # Execute
        return await self._actuator.execute(command)
```

---

### 3. เพิ่ม Actuator Implementations (Infrastructure Layer)

#### 3.1 Robot Arm Actuator

```python
# src/infrastructure/actuators/robot_arm.py
import asyncio
from domain.interfaces.actuator import IActuator, ActuatorCommand

class RobotArmActuator(IActuator):
    """แขนกลหยิบขยะไปวางในถัง"""

    def __init__(self, serial_port: str, baud_rate: int = 115200):
        self._port = serial_port
        self._baud = baud_rate
        self._serial = None

    async def execute(self, command: ActuatorCommand) -> bool:
        """
        สั่งแขนกลหยิบและวาง

        Protocol (ตัวอย่าง):
        - MOVE X Y Z    : เคลื่อนไปตำแหน่ง
        - GRIP          : หยิบ
        - RELEASE       : ปล่อย
        """
        try:
            # 1. คำนวณตำแหน่งจาก position_mm
            pick_pos = self._calculate_pick_position(command.position_mm)

            # 2. คำนวณตำแหน่งถังจาก target_bin
            drop_pos = self._get_bin_position(command.target_bin)

            # 3. ส่งคำสั่ง
            await self._send_command(f"MOVE {pick_pos}")
            await self._send_command("GRIP")
            await self._send_command(f"MOVE {drop_pos}")
            await self._send_command("RELEASE")

            return True

        except Exception as e:
            print(f"Robot arm error: {e}")
            return False

    def _calculate_pick_position(self, position_mm: float) -> str:
        # แปลง mm บนสายพาน → พิกัด XYZ ของแขนกล
        x = position_mm
        y = 0  # สายพานอยู่ตำแหน่ง Y คงที่
        z = 50  # ความสูงหยิบ
        return f"{x} {y} {z}"

    def _get_bin_position(self, bin_number: int) -> str:
        # ตำแหน่งถังแต่ละช่อง
        positions = {
            1: "100 200 100",  # Cardboard
            2: "200 200 100",  # Glass
            3: "300 200 100",  # Metal
            4: "400 200 100",  # Paper
            5: "500 200 100",  # Plastic
            6: "600 200 100",  # Trash
        }
        return positions.get(bin_number, positions[6])
```

#### 3.2 Conveyor Gate Actuator (ง่ายกว่าแขนกล)

```python
# src/infrastructure/actuators/conveyor_gate.py
from domain.interfaces.actuator import IActuator, ActuatorCommand

class ConveyorGateActuator(IActuator):
    """ระบบ Gate เปิด-ปิดตามช่องที่ต้องการ"""

    def __init__(self, gpio_controller):
        self._gpio = gpio_controller
        # GPIO pins สำหรับแต่ละ gate
        self._gate_pins = {
            1: 17,  # Cardboard
            2: 18,  # Glass
            3: 27,  # Metal
            4: 22,  # Paper
            5: 23,  # Plastic
            6: 24,  # Trash
        }

    async def execute(self, command: ActuatorCommand) -> bool:
        """เปิด gate ตามช่องที่กำหนด"""
        pin = self._gate_pins.get(command.target_bin)
        if not pin:
            return False

        # เปิด gate
        self._gpio.output(pin, True)
        await asyncio.sleep(0.5)  # รอให้ขยะตก
        self._gpio.output(pin, False)

        return True
```

---

## Full System Flow (Phase 2)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONVEYOR SYSTEM                                │
│                                                                             │
│  ┌─────────┐                                                   ┌─────────┐ │
│  │  Feed   │                                                   │  Bins   │ │
│  │  Hopper │                                                   │ 1 2 3 4 │ │
│  └────┬────┘                                                   │ 5 6     │ │
│       │                                                        └────┬────┘ │
│       ▼                                                             │      │
│  ════════════════════════════════════════════════════════════════════      │
│  ═══► CONVEYOR BELT (ความเร็วคงที่) ═══════════════════════════════►═══    │
│  ════════════════════════════════════════════════════════════════════      │
│       │              │                           │                         │
│       ▼              ▼                           ▼                         │
│  ┌─────────┐    ┌─────────┐                 ┌─────────┐                    │
│  │ Trigger │    │ Camera  │                 │ Actuator│                    │
│  │ Sensor  │    │ (Vision)│                 │(Arm/Gate)│                    │
│  └────┬────┘    └────┬────┘                 └────┬────┘                    │
│       │              │                           │                         │
└───────┼──────────────┼───────────────────────────┼─────────────────────────┘
        │              │                           │
        ▼              ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SOFTWARE SYSTEM                                │
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Trigger │───▶│ Capture │───▶│   AI    │───▶│  Queue  │───▶│Dispatch │  │
│  │ Event   │    │ Image   │    │ Classify│    │ (Delay) │    │ Action  │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                                     │                             │        │
│                                     ▼                             ▼        │
│                              ClassificationResult          ActuatorCommand │
│                              {                             {               │
│                                category: PLASTIC,            target_bin: 5,│
│                                sub_result: PET,              position: 450 │
│                                confidence: 0.95              }             │
│                              }                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Timing & Queue System

```python
# src/infrastructure/queue/action_queue.py
import asyncio
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class QueuedAction:
    """Action รอดำเนินการ"""
    command: ActuatorCommand
    execute_at: datetime  # เวลาที่ต้อง execute

class ActionQueue:
    """Queue สำหรับ delay action ตามความเร็วสายพาน"""

    def __init__(
        self,
        actuator: IActuator,
        conveyor_speed_mm_per_sec: float = 100,  # 10 cm/s
        camera_to_actuator_mm: float = 500       # 50 cm
    ):
        self._actuator = actuator
        self._speed = conveyor_speed_mm_per_sec
        self._distance = camera_to_actuator_mm
        self._queue: asyncio.Queue = asyncio.Queue()

    def calculate_delay(self, detection_position: float = 0) -> float:
        """คำนวณ delay จากตำแหน่ง camera ถึง actuator"""
        remaining_distance = self._distance - detection_position
        return remaining_distance / self._speed

    async def enqueue(self, command: ActuatorCommand):
        """เพิ่ม action เข้า queue พร้อม delay"""
        delay = self.calculate_delay(command.position_mm)
        execute_at = datetime.now() + timedelta(seconds=delay)

        await self._queue.put(QueuedAction(
            command=command,
            execute_at=execute_at
        ))

    async def process_loop(self):
        """Loop ประมวลผล queue"""
        while True:
            action = await self._queue.get()

            # รอจนถึงเวลา execute
            now = datetime.now()
            if action.execute_at > now:
                wait_seconds = (action.execute_at - now).total_seconds()
                await asyncio.sleep(wait_seconds)

            # Execute
            await self._actuator.execute(action.command)
```

---

## Project Structure (Phase 2 เพิ่มเติม)

```
ai-classification/
├── src/
│   ├── domain/
│   │   ├── interfaces/
│   │   │   ├── classifier.py        # เดิม
│   │   │   ├── image_processor.py   # เดิม
│   │   │   └── actuator.py          # ✅ ใหม่
│   │   └── ...
│   │
│   ├── application/
│   │   ├── classify_trash.py        # เดิม
│   │   ├── routing_service.py       # เดิม
│   │   └── dispatch_action.py       # ✅ ใหม่
│   │
│   ├── infrastructure/
│   │   ├── models/                  # เดิม
│   │   ├── image/                   # เดิม
│   │   ├── actuators/               # ✅ ใหม่
│   │   │   ├── __init__.py
│   │   │   ├── robot_arm.py
│   │   │   ├── conveyor_gate.py
│   │   │   └── mock_actuator.py     # สำหรับ test
│   │   └── queue/                   # ✅ ใหม่
│   │       ├── __init__.py
│   │       └── action_queue.py
│   │
│   └── presentation/
│       └── api/
│           └── routes.py            # เพิ่ม endpoint สำหรับ manual control
```

---

## สรุป: สิ่งที่ต้องทำจริงๆ

### Files ที่ต้องสร้างใหม่:
| File | Lines (ประมาณ) | Complexity |
|------|----------------|------------|
| `domain/interfaces/actuator.py` | ~30 | ง่าย |
| `application/dispatch_action.py` | ~50 | ง่าย |
| `infrastructure/actuators/robot_arm.py` | ~80 | ปานกลาง |
| `infrastructure/actuators/conveyor_gate.py` | ~40 | ง่าย |
| `infrastructure/queue/action_queue.py` | ~60 | ปานกลาง |

**รวม: ~260 lines** (น้อยมากเมื่อเทียบกับ Phase 1)

### Files ที่ไม่ต้องแก้:
- ทุกอย่างใน `domain/entities/`
- ทุกอย่างใน `domain/enums/`
- `ClassifyTrashUseCase`
- `RoutingService`
- `TrashNetClassifier`, `PlasticNetClassifier`
- `IClassifier`, `IImageProcessor`

---

## Hardware Recommendations

### Option A: Gate System (ง่าย, ถูก)
```
สายพานเอียง → Gate เปิด-ปิด → ขยะตกลงถัง

ข้อดี:
- ราคาถูก (~5,000-10,000 บาท)
- ไม่มี moving parts ซับซ้อน
- Maintenance ต่ำ

ข้อเสีย:
- ต้องวางขยะให้ห่างกัน
- Speed จำกัด
```

### Option B: Robot Arm (ซับซ้อน, แม่นยำ)
```
สายพานราบ → แขนกลหยิบ → วางในถัง

ข้อดี:
- รองรับขยะติดกันได้
- Speed สูงกว่า
- ดูเท่!

ข้อเสีย:
- ราคาสูง (~30,000-100,000+ บาท)
- ต้อง calibrate บ่อย
- Maintenance สูง
```

### คำแนะนำ:
**เริ่มจาก Gate System ก่อน** → พิสูจน์ว่า AI ทำงานได้ → ค่อยอัพเป็น Robot Arm

---

## Timeline แนะนำ

```
Phase 1 (ตอนนี้)
├── AI Classification ✅
└── Manual sorting (คนดูจอ + ทิ้งเอง)

Phase 1.5 (ต่อยอดเล็กน้อย)
├── เพิ่ม Position tracking
└── เพิ่ม Actuator interface (mock)

Phase 2a (Gate System)
├── สายพานช้า + Gate
├── ทดสอบ timing
└── ปรับ accuracy

Phase 2b (Robot Arm) - Optional
├── เปลี่ยน actuator implementation
└── Calibrate แขนกล
└── ไม่ต้องแก้ AI เลย!
```

---

## ข้อสรุปสุดท้าย

**คุณคิดถูกแล้วตั้งแต่วันแรก:**

1. **แยก AI ออกจาก Hardware** → ต่อยอดง่าย
2. **Clean Architecture** → เปลี่ยน actuator ไม่กระทบ AI
3. **Interface-based** → Mock ได้, Test ได้, เปลี่ยน hardware ได้

**ความจริงที่สำคัญ:**
> บริษัทที่ล้มเหลว = เอา Hardware นำ Software
> คุณกำลังทำ = Software นำ Hardware (ถูกทาง!)

เมื่อ AI ทำงานได้ดี การเพิ่ม hardware เป็นแค่ "plug adapter เข้าไป"
