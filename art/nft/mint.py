"""Ethereum minting via web3.py."""

from __future__ import annotations

import json
import os
from pathlib import Path

from web3 import Web3


ABI_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "contracts" / "ArtAgent.sol" / "ArtAgent.json"
DEPLOYMENT_PATH = Path(__file__).resolve().parents[2] / "deployment.json"


class Minter:
    """Handles on-chain minting of ArtAgent NFTs."""

    def __init__(
        self,
        rpc_url: str | None = None,
        private_key: str | None = None,
        contract_address: str | None = None,
    ):
        rpc_url = rpc_url or os.environ.get("SEPOLIA_RPC_URL") or os.environ["MAINNET_RPC_URL"]
        self.private_key = private_key or os.environ["DEPLOYER_PRIVATE_KEY"]

        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to RPC: {rpc_url}")

        self.account = self.w3.eth.account.from_key(self.private_key)

        # Load contract ABI
        if not ABI_PATH.exists():
            raise FileNotFoundError(
                f"Contract ABI not found at {ABI_PATH}. Run `npm run compile` first."
            )
        with open(ABI_PATH) as f:
            artifact = json.load(f)

        # Load deployed address
        address = contract_address
        if address is None:
            if not DEPLOYMENT_PATH.exists():
                raise FileNotFoundError(
                    f"deployment.json not found. Deploy the contract first."
                )
            with open(DEPLOYMENT_PATH) as f:
                address = json.load(f)["address"]

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(address),
            abi=artifact["abi"],
        )

    def mint(
        self,
        *,
        to: str | None = None,
        generation: int,
        metadata_uri: str,
    ) -> str:
        """Mint a new ArtAgent NFT.

        Args:
            to: Recipient address (defaults to deployer).
            generation: The evolutionary generation number.
            metadata_uri: IPFS URI for the token metadata (``ipfs://...``).

        Returns:
            Transaction hash as hex string.
        """
        recipient = Web3.to_checksum_address(to) if to else self.account.address

        tx = self.contract.functions.mint(
            recipient, generation, metadata_uri
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
                "gas": 300_000,
                "maxFeePerGas": self.w3.eth.gas_price * 2,
                "maxPriorityFeePerGas": self.w3.to_wei(1, "gwei"),
            }
        )

        signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()

    def wait_for_receipt(self, tx_hash: str, timeout: int = 120) -> dict:
        """Wait for a transaction to be mined and return the receipt."""
        return self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
