const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ArtAgent", function () {
  let contract, owner, other;

  beforeEach(async function () {
    [owner, other] = await ethers.getSigners();
    const ArtAgent = await ethers.getContractFactory("ArtAgent");
    contract = await ArtAgent.deploy(owner.address);
  });

  it("should deploy with correct name and symbol", async function () {
    expect(await contract.name()).to.equal("ArtAgent");
    expect(await contract.symbol()).to.equal("ARTAGENT");
  });

  it("should mint with correct URI and generation", async function () {
    const uri = "ipfs://QmTest123";
    await contract.mint(owner.address, 42, uri);

    expect(await contract.totalSupply()).to.equal(1);
    expect(await contract.tokenURI(0)).to.equal(uri);
    expect(await contract.tokenGeneration(0)).to.equal(42);
    expect(await contract.ownerOf(0)).to.equal(owner.address);
  });

  it("should emit GenerationMinted event", async function () {
    const uri = "ipfs://QmTest456";
    await expect(contract.mint(owner.address, 7, uri))
      .to.emit(contract, "GenerationMinted")
      .withArgs(0, 7, uri);
  });

  it("should increment token IDs", async function () {
    await contract.mint(owner.address, 1, "ipfs://a");
    await contract.mint(owner.address, 2, "ipfs://b");
    await contract.mint(other.address, 3, "ipfs://c");

    expect(await contract.totalSupply()).to.equal(3);
    expect(await contract.ownerOf(2)).to.equal(other.address);
    expect(await contract.tokenGeneration(2)).to.equal(3);
  });

  it("should reject minting from non-owner", async function () {
    await expect(
      contract.connect(other).mint(other.address, 1, "ipfs://x")
    ).to.be.revertedWithCustomError(contract, "OwnableUnauthorizedAccount");
  });
});
