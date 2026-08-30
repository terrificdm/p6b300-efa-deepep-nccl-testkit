#!/usr/bin/env python3
"""生成 1 台 p6-b300.48xlarge（17 网卡 / 16 EFA）的 LaunchTemplateData JSON。

布局：card0/device0 = ENA（不写 InterfaceType，承载 IP；该机型主卡不支持 EFA，
写 "efa" 会被 run-instances 拒绝），card1..16/device0 = efa-only（纯 EFA，不占 IPv4）。
不设 Placement Group（CB 不支持），不自动分配公网 IP（多 ENI 不支持）。
与 p5en 版的差异就在布局：p5en 是 16 卡（card0=efa + 15 efa-only），
b300 是 17 卡（card0=ENA + 16 efa-only）——这是两个机型唯一不能靠改数字迁移的地方。
改编自 terrificdm/p6b300-efa-deepepv2-test-runbook（Apache-2.0）。
"""
import argparse
import json
from pathlib import Path


def build(args):
    # card0：主网卡，ENA。不写 InterfaceType 即默认 ENA；这里显式写 "interface"
    # 也不行（LaunchTemplateData 只接受 efa/efa-only 或省略），所以省略。
    interfaces = [{
        "NetworkCardIndex": 0,
        "DeviceIndex": 0,
        "SubnetId": args.subnet_id,
        "Groups": [args.security_group_id],
        "DeleteOnTermination": True,
    }]
    for index in range(1, 17):
        interfaces.append({
            "NetworkCardIndex": index,
            "DeviceIndex": 0,
            "InterfaceType": "efa-only",
            "SubnetId": args.subnet_id,
            "Groups": [args.security_group_id],
            "DeleteOnTermination": True,
        })
    data = {
        "ImageId": args.ami_id,
        "InstanceType": "p6-b300.48xlarge",
        "KeyName": args.key_name,
        "NetworkInterfaces": interfaces,
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": args.root_volume_gib,
                "VolumeType": "gp3",
                "Iops": 3000,
                "Throughput": 125,
                "Encrypted": True,
                "DeleteOnTermination": True,
            },
        }],
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpTokens": "required",
            "HttpPutResponseHopLimit": 2,
        },
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": args.name},
                {"Key": "Workload", "Value": "p6b300-deepep-nccl-test"},
                {"Key": "CapacityBlock", "Value": args.capacity_block_id},
            ],
        }, {
            "ResourceType": "volume",
            "Tags": [
                {"Key": "Name", "Value": f"{args.name}-root"},
                {"Key": "Workload", "Value": "p6b300-deepep-nccl-test"},
            ],
        }],
    }
    # 自检：布局错了宁可不出文件
    assert len(interfaces) == 17
    assert sum(x.get("InterfaceType") == "efa-only" for x in interfaces) == 16
    assert not any(x.get("InterfaceType") == "efa" for x in interfaces)
    assert "InterfaceType" not in interfaces[0]
    assert {x["NetworkCardIndex"] for x in interfaces} == set(range(17))
    assert "Placement" not in data
    assert not any("AssociatePublicIpAddress" in x for x in interfaces)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ami-id", required=True)
    p.add_argument("--subnet-id", required=True)
    p.add_argument("--security-group-id", required=True)
    p.add_argument("--key-name", required=True)
    p.add_argument("--capacity-block-id", required=True)
    p.add_argument("--root-volume-gib", type=int, default=500)
    p.add_argument("--name", default="p6b300-deepep-nccl")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.root_volume_gib < 200:
        p.error("root volume must be at least 200 GiB")
    data = build(args)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {path}: 17 interfaces (1 ENA + 16 efa-only), "
          f"root {args.root_volume_gib} GiB, no PG, no auto public IP")


if __name__ == "__main__":
    main()
