from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from random import randint

from locust import HttpUser, between, task


JPEG_1X1 = bytes.fromhex(
    "ffd8ffe000104a46494600010101006000600000ffdb004300"
    "0302020302020303030304030304050805050404050a07070608"
    "0c0a0c0c0b0a0b0b0d0e12100d0e110e0b0b10161011131415"
    "15150c0f171816141812141514ffdb0043010304040504050905"
    "0509140d0b0d1414141414141414141414141414141414141414"
    "1414141414141414141414141414141414141414141414141414"
    "141414141414ffc00011080001000103012200021101031101ff"
    "c40014000100000000000000000000000000000000000008ffc4"
    "0014100100000000000000000000000000000000000000ffda00"
    "0c03010002110311003f00b2c001ffd9"
)


class AttendanceUploadUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task
    def upload_attendance_image(self) -> None:
        classroom = randint(1, 300)
        data = {
            "camera_id": f"classroom-{classroom}",
            "class_id": f"class-{classroom}",
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        files = {
            "image": (
                "sample.jpg",
                BytesIO(JPEG_1X1),
                "image/jpeg",
            )
        }
        self.client.post("/attendance/capture", data=data, files=files, name="/attendance/capture")
